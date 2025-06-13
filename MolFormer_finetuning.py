import sys
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import pandas as pd
import numpy as np
from itertools import product

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
script_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    "molformer": "ibm/MoLFormer-XL-both-10pct",
    "molformer-finetuned": "ibm/MoLFormer-XL-both-10pct"
}

FINETUNED_PATH = os.path.join(script_dir, "MoLFormer_finetuned_model.pth")

TASK_CONFIGS = {
    "peptides": {
        "data_path": os.path.join(script_dir, "data", "downstream_task_data", "Peptides_CV"),
        "file_prefix": "peptides",
        "target": "Permeability_Label",
        "metric": roc_auc_score,
        "metric_name": "AUC-ROC"
    },
    "anti_cancer": {
        "data_path": os.path.join(script_dir, "data", "downstream_task_data", "Anti_Cancer_CV"),
        "file_prefix": "cancer_activity",
        "target": "Activity_Label",
        "metric": matthews_corrcoef,
        "metric_name": "MCC"
    }
}

def get_data_file_path(task, fold, split, data_split="rd"):
    """
    Constructs the file path for a specific data split (train/val/test) and fold of a given task.

    Args:
        task (str): Task name (e.g., "peptides" or "anti_cancer").
        fold (int): Fold number.
        split (str): Data split ("train", "val", or "test").
        data_split (str): Dataset variant ("sf" or "rd").

    Returns:
        str: Full path to the corresponding CSV file.
    """
    task_props = TASK_CONFIGS[task]
    file_name = f"{task_props['file_prefix']}_fold{fold}_{data_split}_{split}.csv"
    return os.path.join(task_props['data_path'], file_name)

class PropertyDataset(Dataset):
    """
    A PyTorch Dataset for molecular property prediction using SMILES strings.

    Args:
        data (pd.DataFrame): DataFrame containing SMILES and target values.
        tokenizer (AutoTokenizer): HuggingFace tokenizer.
        target_column (str): Column name containing the label.
        max_length (int): Maximum sequence length for tokenization.
    """
    def __init__(self, data, tokenizer, target_column, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.target_column = target_column
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]['Standardized_SMILES']
        label = self.data.iloc[idx][self.target_column]
        label_tensor = torch.tensor(label, dtype=torch.float32)
        encoding = self.tokenizer(smiles, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label_tensor
        }

def collate_batch(batch):
    """
    Custom collate function for padding a batch of SMILES token sequences and stacking labels.

    Args:
        batch (list): List of dictionaries from PropertyDataset.__getitem__.

    Returns:
        dict: Dictionary containing batched input_ids, attention_mask, and labels tensors.
    """
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]
    batch_padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    attention_mask = (batch_padded_inputs != tokenizer.pad_token_id).float().to(device)
    labels = torch.stack(labels).to(device)
    return {
        'input_ids': batch_padded_inputs,
        'attention_mask': attention_mask,
        'labels': labels
    }

class MolFormerForSequenceClassification(nn.Module):
    """
    A sequence classification model based on MoLFormer with a classification head.

    Args:
        base_model_name (str): Name of the pretrained MoLFormer model.
        num_labels (int): Number of output labels (1 for binary classification).
    """
    def __init__(self, base_model_name, num_labels):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            base_model_name, num_labels=num_labels, deterministic_eval=True, trust_remote_code=True
        ).to(device)
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.base_model.config.hidden_size, num_labels)
        ).to(device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        logits = self.classification_head(outputs.last_hidden_state[:, 0, :])
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits.squeeze(-1), labels.float().to(device))
        return {"loss": loss, "logits": logits}

def load_data(task, fold, split, data_split="rd"):
    """
    Loads a CSV file for a specific task, fold, and split.

    Args:
        task (str): Task name.
        fold (int): Fold number.
        split (str): Data split ("train", "val", or "test").
        data_split (str): Dataset splitting methods ("sf" or "rd").

    Returns:
        pd.DataFrame: Loaded data.
    """
    path = get_data_file_path(task, fold, split, data_split)
    return pd.read_csv(path)

def hyperparameter_search(model_type, task, data_split="rd", num_folds=5):
    """
    Performs a grid search over learning rates and batch sizes using cross-validation.

    Args:
        model_type (str): Either "molformer" or "molformer-finetuned".
        task (str): Task name.
        data_split (str): Dataset splitting methods.
        num_folds (int): Number of cross-validation folds.

    Returns:
        dict: Best hyperparameter configuration with keys 'learning_rate' and 'batch_size'.
    """
    print(f"Starting hyperparameter search for {model_type.upper()} on {task.upper()}")
    base_model = MODEL_CONFIGS[model_type]
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    learning_rates = [1e-4, 1e-5, 5e-5]
    batch_sizes = [8, 16]
    best_score, best_config = -1.0, None

    for lr, bs in product(learning_rates, batch_sizes):
        fold_scores = []
        for fold in range(1, num_folds + 1):
            train_df = load_data(task, fold, 'train', data_split)
            val_df = load_data(task, fold, 'val', data_split)
            train_ds = PropertyDataset(train_df, tokenizer, TASK_CONFIGS[task]['target'])
            val_ds = PropertyDataset(val_df, tokenizer, TASK_CONFIGS[task]['target'])
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_batch)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_batch)

            model = MolFormerForSequenceClassification(base_model, 1)
            if model_type == "molformer-finetuned":
                state_dict = torch.load(FINETUNED_PATH, map_location=device)
                new_state_dict = {k if k.startswith("base_model.") else "base_model." + k: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
            optimizer = AdamW(model.parameters(), lr=lr)

            for _ in range(3):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    out = model(**batch)
                    out['loss'].backward()
                    optimizer.step()

            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    out = model(**batch)
                    all_logits.extend(out['logits'].cpu().numpy().flatten())
                    all_labels.extend(batch['labels'].cpu().numpy())

            prob = torch.sigmoid(torch.tensor(all_logits)).numpy()
            pred = (prob > 0.5).astype(int)
            metric_func = TASK_CONFIGS[task]['metric']
            score = metric_func(all_labels, pred if metric_func == matthews_corrcoef else prob)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_config = {'learning_rate': lr, 'batch_size': bs}

    print("Best Hyperparameters:", best_config)
    return best_config

def finetune(model_type, task, config, data_split="rd"):
    """
    Fine-tunes the MoLFormer model using the best hyperparameters on the specified task.

    Args:
        model_type (str): "molformer" or "molformer-finetuned".
        task (str): Task name.
        config (dict): Dictionary containing 'learning_rate' and 'batch_size'.
        data_split (str): Dataset splitting methods.
    """
    print(f"Starting finetuning with best config for {model_type.upper()} on {task.upper()}: {config}")
    base_model = MODEL_CONFIGS[model_type]
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    all_fold_scores = []
    print(f"Using metric: {TASK_CONFIGS[task]['metric_name']}")

    for fold in range(1, 6):
        train_df = load_data(task, fold, 'train', data_split)
        val_df = load_data(task, fold, 'val', data_split)
        test_df = load_data(task, fold, 'test', data_split)

        train_ds = PropertyDataset(train_df, tokenizer, TASK_CONFIGS[task]['target'])
        val_ds = PropertyDataset(val_df, tokenizer, TASK_CONFIGS[task]['target'])
        test_ds = PropertyDataset(test_df, tokenizer, TASK_CONFIGS[task]['target'])

        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_batch)
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_batch)

        run_scores = []
        for run in range(1, 6):
            model = MolFormerForSequenceClassification(base_model, 1)
            if model_type == "molformer-finetuned":
                state_dict = torch.load(FINETUNED_PATH, map_location=device)
                new_state_dict = {k if k.startswith("base_model.") else "base_model." + k: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
            optimizer = AdamW(model.parameters(), lr=config['learning_rate'])

            best_loss, patience, counter, best_state = float('inf'), 5, 0, None

            for epoch in range(50):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    out = model(**batch)
                    out['loss'].backward()
                    optimizer.step()

                val_loss = 0
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        out = model(**batch)
                        val_loss += out['loss'].item()
                val_loss /= len(val_loader)

                if val_loss < best_loss:
                    best_loss, counter, best_state = val_loss, 0, model.state_dict()
                else:
                    counter += 1
                if counter >= patience:
                    break

            if best_state:
                model.load_state_dict(best_state)

            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    out = model(**batch)
                    all_logits.extend(out['logits'].cpu().numpy().flatten())
                    all_labels.extend(batch['labels'].cpu().numpy())

            prob = torch.sigmoid(torch.tensor(all_logits)).numpy()
            pred = (prob > 0.5).astype(int)
            metric_func = TASK_CONFIGS[task]['metric']
            score = metric_func(all_labels, pred if metric_func == matthews_corrcoef else prob)
            run_scores.append(score)
            print(f"Fold {fold}, Run {run}, Score: {score:.4f}")

        fold_avg = np.mean(run_scores)
        all_fold_scores.append(fold_avg)
        print(f"Fold {fold} Mean Score: {fold_avg:.4f}")

    overall_mean = np.mean(all_fold_scores)
    overall_std = np.std(all_fold_scores)
    overall_se = overall_std / np.sqrt(len(all_fold_scores))

    print(f"\nFinal Cross-Validation Results for {task.upper()} using {TASK_CONFIGS[task]['metric_name']}")
    print(f"Mean Score: {overall_mean:.4f}")
    print(f"Standard Deviation: {overall_std:.4f}")
    print(f"Standard Error: {overall_se:.4f}")

def run_molformer(config):
    """
    Runs the full pipeline: hyperparameter search followed by fine-tuning and evaluation.

    Args:
        config (dict): Configuration dictionary with keys: 'model_type', 'sub_task', 'data_split'.
    """
    model_type = config.get("model_type", "molformer")
    task = config.get("sub_task", "peptides")
    data_split = config.get("data_split", "rd")
    best_config = hyperparameter_search(model_type, task, data_split)
    finetune(model_type, task, best_config, data_split)

if __name__ == "__main__":
    config = {"model_type": "molformer", "sub_task": "peptides", "data_split": "rd"}
    run_molformer(config)
