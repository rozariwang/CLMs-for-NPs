import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import torch
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch import nn
from tqdm import tqdm
import os
import json
import pandas as pd
from huggingface_hub import login
from itertools import product
from tokenisers import AISTokenizer

wandb.login(key="your_wandb_api_key")  # Replace with your actual Wandb API key   
login(token="your_huggingface_token")  # Replace with your actual Hugging Face API token

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer_path = './hhwang/shiawaseda/Datasets/ais_vocab.json'
tokenizer = AISTokenizer(tokenizer_path)
base_model = "rozariwang/GPT-AIS-rds"
config = wandb.config


def load_crossval_data(subtask, fold):
    """
    Load train, validation, and test data for a specific fold of a subtask.
    """
    train_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Tox21_CV/{subtask}_fold{fold}_rd_train.csv')
    val_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Tox21_CV/{subtask}_fold{fold}_rd_val.csv')
    test_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Tox21_CV/{subtask}_fold{fold}_rd_test.csv')
    return train_df, val_df, test_df

class Tox21Dataset(Dataset):
    def __init__(self, data, tokenizer, target_column, max_length=512):
        self.data = data.dropna(subset=[target_column])
        self.tokenizer = tokenizer
        self.target_column = target_column
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['standardized_smiles']
        label = self.data.iloc[idx][self.target_column]
        # Use encode method explicitly
        tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class GPT2ForSequenceClassification(nn.Module):
    def __init__(self, base_model_name, num_labels, pos_weight):
        super(GPT2ForSequenceClassification, self).__init__()
        self.base_model = GPT2Model.from_pretrained(base_model_name).to(device)
        self.num_labels = num_labels
        self.pos_weight = torch.tensor([pos_weight]).to(device)  # Weight for positive class

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.base_model.config.hidden_size, num_labels)
        ).to(device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None

        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]
        
        attention_mask = attention_mask.unsqueeze(-1)  # Shape: [batch_size, seq_length, 1]
        # Broadcast attention_mask to match hidden_states' shape
        attention_mask = attention_mask.expand_as(hidden_states)
        
        ## Max Pooling ##
        masked_hidden_states = hidden_states.masked_fill(attention_mask == 0, -float('inf'))
        pooled_output, _ = masked_hidden_states.max(dim=1)  # Max pooling across seq_length

        logits = self.classification_head(pooled_output)  # Shape: [batch_size, num_labels]

        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fn(logits.squeeze(-1), labels)

        return {"loss": loss, "logits": logits}
    

def collate_batch(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    batch_padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.vocab['[PAD]']).to(device)
    attention_mask = (batch_padded_inputs != tokenizer.vocab['[PAD]']).float().to(device)
    labels = torch.tensor(labels, dtype=torch.float).to(device)
    return {
        'input_ids': batch_padded_inputs,
        'attention_mask': attention_mask,
        'labels': labels
    }
    
    
def crossval_hyperparameter_search(subtasks, num_folds=5):
    """
    Perform cross-validation-based hyperparameter search for each subtask.
    """
    tokenizer = AISTokenizer(tokenizer_path)

    learning_rates = [1e-4, 1e-5, 5e-5]
    batch_sizes = [8, 16]

    results = {}

    for subtask in subtasks:
        print(f"Starting cross-validation for subtask: {subtask}")

        best_auc = 0.0
        best_config = None

        for lr, batch_size in product(learning_rates, batch_sizes):
            print(f"Testing config: LR={lr}, Batch Size={batch_size}")

            fold_aucs = []

            for fold in range(1, num_folds + 1):
                # Load data for the current fold
                train_df, val_df, _ = load_crossval_data(subtask, fold)

                train_dataset = Tox21Dataset(train_df, tokenizer, target_column=subtask)
                val_dataset = Tox21Dataset(val_df, tokenizer, target_column=subtask)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

                num_positives = len(train_dataset.data[train_dataset.data[subtask] == 1])
                num_negatives = len(train_dataset.data[train_dataset.data[subtask] == 0])
                pos_weight = num_negatives / num_positives if num_positives > 0 else 1

                # Initialize model and optimizer
                model = GPT2ForSequenceClassification(base_model, num_labels=1, pos_weight=pos_weight)
                optimizer = AdamW(model.parameters(), lr=lr)

                # Train the model for 3 epochs
                for epoch in range(3):
                    model.train()
                    for batch in train_loader:
                        optimizer.zero_grad()
                        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                        loss = outputs['loss']
                        loss.backward()
                        optimizer.step()

                # Validate the model
                model.eval()
                val_targets = []
                val_predictions = []

                with torch.no_grad():
                    for batch in val_loader:
                        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                        logits = outputs['logits'].squeeze(-1)
                        predictions = torch.sigmoid(logits).cpu().numpy()  # Convert logits to probabilities

                        val_targets.extend(batch['labels'].cpu().numpy())
                        val_predictions.extend(predictions)

                # Calculate AUC-ROC for the current fold
                auc = roc_auc_score(val_targets, val_predictions)
                fold_aucs.append(auc)

            # Calculate average AUC-ROC across folds
            avg_auc = np.mean(fold_aucs)
            print(f"Config LR={lr}, Batch Size={batch_size}, Avg AUC-ROC={avg_auc:.4f}")

            # Update best configuration if this is the best so far
            if avg_auc > best_auc:
                best_auc = avg_auc
                best_config = {'learning_rate': lr, 'batch_size': batch_size}

        results[subtask] = {
            'best_auc': best_auc,
            'best_config': best_config
        }

    return results


if __name__ == "__main__":
    subtasks = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]
    results = crossval_hyperparameter_search(subtasks)
    print("Cross-Validation Results (AUC-ROC):", results)