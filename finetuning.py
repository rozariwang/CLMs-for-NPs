# Combined Fine-tuning Script for Multiple Downstream Tasks
import sys
import os
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model, AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch import nn
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tokenisers import NPBPETokenizer, CharLevelTokenizer, AISTokenizer

def run_finetuning(config):
    """
    Run fine-tuning and evaluation for multiple downstream classification tasks 
    using pretrained language models (GPT, Mamba, or Mamba2) and 8 different tokenizers.

    Args:
        config (dict): Configuration dictionary with keys:
            - 'sub_task' (str): One of ['anti_cancer', 'peptides', 'tastes'].
            - 'data_split' (str, optional): Suffix indicating downstream task data split format ('sf' or 'rd'). Default is 'rd'.
            - 'model_split' (str, optional): Suffix used in model name construction, indicating pre-training data split. Default is 'rds'.

    Raises:
        ValueError: If an unknown sub-task is specified.

    Behavior:
        - Loads the appropriate tokenizer and model for each task.
        - Fine-tunes models using 5-fold cross-validation with early stopping.
        - Evaluates on test sets and logs performance metrics (e.g., MCC, AUC).
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_dir = os.path.join(script_dir, "vocab_files")
    
    downstream_tasks = {
        "anti_cancer": {
            "data_path": os.path.join(script_dir, "data", "downstream_task_data", "Anti_Cancer_CV"),
            "file_prefix": "cancer_activity",
            "target": "Activity_Label",
            "metric": matthews_corrcoef,
            "label_type": float,
            "num_labels": 1,
            "loss_type": "bce",
            "use_class_weights": False
        },
        "peptides": {
            "data_path": os.path.join(script_dir, "data", "downstream_task_data", "Peptides_CV"),
            "file_prefix": "peptides",
            "target": "Permeability_Label",
            "metric": roc_auc_score,
            "label_type": float,
            "num_labels": 1,
            "loss_type": "bce",
            "use_class_weights": False
        },
        "tastes": {
            "data_path": os.path.join(script_dir, "data", "downstream_task_data", "4Tastes_CV"),
            "file_prefix": "4Tastes",
            "target": "Taste_Label",
            "metric": matthews_corrcoef,
            "label_type": int,
            "num_labels": 4,
            "loss_type": "crossentropy",
            "use_class_weights": True
        }
    }


    task = config["sub_task"]
    if task not in downstream_tasks:
        raise ValueError(f"Unknown sub_task: {task}")
    props = downstream_tasks[task]
    data_suffix = config.get("data_split", "rd")
    model_suffix = config.get("model_split", "rds")   # for model name

    tokenizer_classes = {
        "char": CharLevelTokenizer,
        "bpe": AutoTokenizer.from_pretrained,
        "ais": AISTokenizer,
        "npbpe60": NPBPETokenizer,
        "npbpe100": NPBPETokenizer,
        "npbpe1000": NPBPETokenizer,
        "npbpe7924": NPBPETokenizer,
        "npbpe30k": NPBPETokenizer
    }

    tokenizer_paths = {
        "char": os.path.join(vocab_dir, 'vocab.json'),
        "bpe": "seyonec/PubChem10M_SMILES_BPE_450k",
        "ais": os.path.join(vocab_dir, 'ais_vocab.json'),
        "npbpe60": os.path.join(vocab_dir, 'npbpe_60.json'),
        "npbpe100": os.path.join(vocab_dir, 'npbpe_100.json'),
        "npbpe1000": os.path.join(vocab_dir, 'npbpe_1000.json'),
        "npbpe7924": os.path.join(vocab_dir, 'npbpe_7924vocab.json'),
        "npbpe30k": os.path.join(vocab_dir, 'npbpe_tokenizer.json')
    }


    model_names = [
        f"rozariwang/{arch}-{tok}-{model_suffix}"
        for arch in ["GPT", "M1", "M2"]
        for tok in ["Char", "BPE", "AIS", "npbpe60", "npbpe100", "npbpe1000", "npbpe7924", "npbpe30k"]
    ]

    class SimpleDataset(Dataset):
        """
        Custom PyTorch Dataset for handling SMILES-based classification data.

        Args:
            data (pd.DataFrame): Input DataFrame with SMILES strings and labels.
            tokenizer: Tokenizer used to encode SMILES strings.
            target_column (str): Column name for labels.
            label_type (type): Data type for labels (e.g., float, int).
            max_length (int, optional): Maximum token sequence length. Default is 512.
        """
        def __init__(self, data, tokenizer, target_column, label_type, max_length=512):
            self.data = data
            self.tokenizer = tokenizer
            self.target_column = target_column
            self.max_length = max_length
            self.label_type = label_type

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text = self.data.iloc[idx]['Standardized_SMILES']
            label = self.data.iloc[idx][self.target_column]
            tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, truncation=True)
            return {'input_ids': torch.tensor(tokens, dtype=torch.long), 'labels': torch.tensor(label, dtype=self.label_type)}

    class MambaForSequenceClassificationFull(nn.Module):
        """
        Mamba-based sequence classification model with a classification head.

        Args:
            base_model_name (str): Pretrained model name or path.
            num_labels (int): Number of output labels.
            loss_type (str): 'bce' or 'crossentropy' for loss function.
            class_weights (torch.Tensor, optional): Optional weights for class imbalance.

        Forward Args:
            input_ids (torch.Tensor): Tokenized input sequences.
            attention_mask (torch.Tensor): Attention masks.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            dict: Dictionary with keys 'loss' and 'logits'.
        """
        def __init__(self, base_model_name, num_labels, loss_type='bce', class_weights=None):
            super().__init__()
            self.base_model = MambaLMHeadModel.from_pretrained(base_model_name).to(device)
            self.base_model.lm_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.base_model.config.d_model, num_labels)).to(device)
            self.loss_type = loss_type
            self.class_weights = class_weights

        def forward(self, input_ids, attention_mask=None, labels=None):
            hidden_states = self.base_model.backbone(input_ids.to(device))
            attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states) if attention_mask is not None else None
            masked = hidden_states.masked_fill(attention_mask == 0, -float('inf')) if attention_mask is not None else hidden_states
            pooled_output, _ = masked.max(dim=1)
            logits = self.base_model.lm_head(pooled_output)
            loss = None
            if labels is not None:
                labels = labels.to(device)
                if self.loss_type == 'crossentropy':
                    loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
                    loss = loss_fn(logits, labels)
                else:
                    loss_fn = nn.BCEWithLogitsLoss()
                    loss = loss_fn(logits.squeeze(-1), labels)
            return {"loss": loss, "logits": logits}

    class GPT2ForSequenceClassification(nn.Module):
        """
        GPT2-based sequence classification model with a classification head.

        Args:
            base_model_name (str): Pretrained model name or path.
            num_labels (int): Number of output labels.
            loss_type (str): 'bce' or 'crossentropy' for loss function.
            class_weights (torch.Tensor, optional): Optional weights for class imbalance.

        Forward Args:
            input_ids (torch.Tensor): Tokenized input sequences.
            attention_mask (torch.Tensor): attention masks.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            dict: Dictionary with keys 'loss' and 'logits'.
        """
        def __init__(self, base_model_name, num_labels, loss_type='bce', class_weights=None):
            super().__init__()
            self.base_model = GPT2Model.from_pretrained(base_model_name).to(device)
            self.classification_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.base_model.config.hidden_size, num_labels)).to(device)
            self.loss_type = loss_type
            self.class_weights = class_weights

        def forward(self, input_ids, attention_mask=None, labels=None):
            outputs = self.base_model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            hidden_states = outputs.last_hidden_state
            attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            pooled_output, _ = hidden_states.masked_fill(attention_mask == 0, -float('inf')).max(dim=1)
            logits = self.classification_head(pooled_output)
            loss = None
            if labels is not None:
                labels = labels.to(device)
                if self.loss_type == 'crossentropy':
                    loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
                    loss = loss_fn(logits, labels)
                else:
                    loss_fn = nn.BCEWithLogitsLoss()
                    loss = loss_fn(logits.squeeze(-1), labels)
            return {"loss": loss, "logits": logits}

    def collate_batch(batch, tokenizer_type, tokenizer):
        """
        Custom collation function for DataLoader that pads sequences and generates attention masks.

        Args:
            batch (list of dicts): List of samples from the dataset.
            tokenizer_type (str): Type of tokenizer to determine padding token.
            tokenizer: Tokenizer instance to retrieve padding ID.

        Returns:
            dict: Batch dictionary with keys:
                - 'input_ids': Padded token IDs.
                - 'attention_mask': Binary mask of non-padding tokens.
                - 'labels': Tensor of labels.
        """
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        if tokenizer_type in ["char", "ais"]:
            pad_val = tokenizer.vocab['[PAD]']
        elif tokenizer_type == "bpe":
            pad_val = tokenizer.pad_token_id
        else:
            pad_val = tokenizer.tokenizer.token_to_id('[PAD]')
        padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_val).to(device)
        mask = (padded != pad_val).float().to(device)
        return {'input_ids': padded, 'attention_mask': mask, 'labels': torch.tensor(labels).to(device)}   
   
    for task, props in downstream_tasks.items():
        task = config["sub_task"]
        if task not in downstream_tasks:
            raise ValueError(f"Unknown sub_task: {task}")
        props = downstream_tasks[task]
        for model_name in model_names:
            tokenizer_type = model_name.split("-")[1].lower()
            tokenizer_class = tokenizer_classes[tokenizer_type]
            tokenizer_path = tokenizer_paths[tokenizer_type]
            tokenizer = tokenizer_class(tokenizer_path) if tokenizer_type != 'bpe' else AutoTokenizer.from_pretrained(tokenizer_path)

            target_col = props['target']
            metric = props['metric']
            label_type = torch.float if props['label_type'] == float else torch.long
            num_labels = props['num_labels']

            print(f"\n\n[INFO] Task={task} | Model={model_name} | Tokenizer={tokenizer_type}")
            fold_scores = []
            for fold in range(1, 6):
                run_scores = []
                for run in range(1, 6):
                    file_prefix = props['file_prefix']
                    path = props['data_path']
                    train_df = pd.read_csv(os.path.join(path, f"{file_prefix}_fold{fold}_{data_suffix}_train.csv"))
                    val_df = pd.read_csv(os.path.join(path, f"{file_prefix}_fold{fold}_{data_suffix}_val.csv"))
                    test_df = pd.read_csv(os.path.join(path, f"{file_prefix}_fold{fold}_{data_suffix}_test.csv"))

                    train_ds = SimpleDataset(train_df, tokenizer, target_col, label_type)
                    val_ds = SimpleDataset(val_df, tokenizer, target_col, label_type)
                    test_ds = SimpleDataset(test_df, tokenizer, target_col, label_type)

                    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=lambda x: collate_batch(x, tokenizer_type, tokenizer))
                    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=lambda x: collate_batch(x, tokenizer_type, tokenizer))
                    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=lambda x: collate_batch(x, tokenizer_type, tokenizer))

                    if props.get("use_class_weights", False):
                        all_train_labels = train_df[props["target"]].to_numpy()
                        class_weights = compute_class_weight(
                            class_weight='balanced',
                            classes=np.unique(all_train_labels),
                            y=all_train_labels
                        )
                        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
                    else:
                        class_weights = None
        
                    model_cls = GPT2ForSequenceClassification if 'GPT' in model_name else MambaForSequenceClassificationFull
                    loss_type = props['loss_type']
                    model = model_cls(model_name, num_labels, loss_type=loss_type, class_weights=class_weights)
                    optimizer = AdamW(model.parameters(), lr=1e-4)

                    best_val_loss = float('inf')
                    patience = 5
                    best_model_state = None
                    no_improve_epochs = 0

                    for epoch in range(25):
                        model.train()
                        for batch in train_loader:
                            optimizer.zero_grad()
                            out = model(**batch)
                            out['loss'].backward()
                            optimizer.step()

                        # Validation
                        model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for batch in val_loader:
                                out = model(**batch)
                                val_loss += out['loss'].item()

                        val_loss /= len(val_loader)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_state = model.state_dict()
                            no_improve_epochs = 0
                        else:
                            no_improve_epochs += 1

                        if no_improve_epochs >= patience:
                            print(f"Early stopping at epoch {epoch+1} for fold {fold}, run {run}")
                            break

                    if best_model_state:
                        model.load_state_dict(best_model_state)

                    # Test Evaluation
                    model.eval()
                    all_labels, all_logits = [], []

                    with torch.no_grad():
                        for batch in test_loader:
                            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                            logits = outputs['logits']

                            all_labels.extend(batch['labels'].cpu().numpy())
                            all_logits.extend(logits.cpu().numpy())

                    # Convert to arrays
                    all_logits = np.array(all_logits)
                    all_labels = np.array(all_labels)

                    if task == 'tastes':
                        # CrossEntropy: multiclass classification
                        pred = all_logits.argmax(axis=1)
                        score = matthews_corrcoef(all_labels, pred)

                    else:
                        # BCEWithLogits: binary classification
                        prob = torch.sigmoid(torch.tensor(all_logits)).numpy()
                        pred = (prob > 0.5).astype(int)

                        if props['metric'] == matthews_corrcoef:
                            score = matthews_corrcoef(all_labels, pred)
                        elif props['metric'] == roc_auc_score:
                            score = roc_auc_score(all_labels, prob)

                    print(f"[RESULT] {task} | {model_name} | Fold={fold} | Run={run} | Score={score:.4f}")
                    run_scores.append(score)

                fold_mean = np.mean(run_scores)
                fold_scores.append(fold_mean)
                print(f"[INFO] Fold {fold} Mean Score: {fold_mean:.4f}")

            overall_mean = np.mean(fold_scores)
            overall_std = np.std(fold_scores)
            overall_se = overall_std / np.sqrt(len(fold_scores))
            print(f"\nFinal Results for {task} | {model_name}: Mean={overall_mean:.4f}, Std={overall_std:.4f}, SE={overall_se:.4f}")
