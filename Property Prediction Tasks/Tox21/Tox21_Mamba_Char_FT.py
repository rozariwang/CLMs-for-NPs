import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import torch
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch import nn
from tqdm import tqdm
import os
import pandas as pd
from huggingface_hub import login
from tokenisers import CharLevelTokenizer
import matplotlib.pyplot as plt
from math import sqrt

wandb.login(key="your_wandb_api_key")  # Replace with your actual Wandb API key   
login(token="your_huggingface_token")  # Replace with your actual Hugging Face API token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer_path = './hhwang/shiawaseda/Datasets/vocab.json'
tokenizer = CharLevelTokenizer(tokenizer_path)
base_model = "rozariwang/M1-Char-rds"
print('Training with rozariwang/M1-Char-rds...')

def load_crossval_data(subtask, fold):
    """
    Load train, validation, and test data for a specific fold of a subtask.
    """
    train_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Tox21_CV/{subtask}_fold{fold}_rd_train.csv')
    val_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Tox21_CV/{subtask}_fold{fold}_rd_val.csv')
    test_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Tox21_CV/{subtask}_fold{fold}_rd_test.csv')
    return train_df, val_df, test_df

subtasks = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

best_hyperparams = {
    "NR-AR": {"learning_rate": 1e-05, "batch_size": 16},
    "NR-AR-LBD": {"learning_rate": 5e-05, "batch_size": 8},
    "NR-AhR": {"learning_rate": 0.0001, "batch_size": 8},
    "NR-Aromatase": {"learning_rate": 5e-05, "batch_size": 16},
    "NR-ER": {"learning_rate": 1e-05, "batch_size": 16},
    "NR-ER-LBD": {"learning_rate": 0.0001, "batch_size": 8},
    "NR-PPAR-gamma": {"learning_rate": 0.0001, "batch_size": 8},
    "SR-ARE": {"learning_rate": 5e-05, "batch_size": 8},
    "SR-ATAD5": {"learning_rate": 0.0001, "batch_size": 16},
    "SR-HSE": {"learning_rate": 0.0001, "batch_size": 8},
    "SR-MMP": {"learning_rate": 5e-05, "batch_size": 16},
    "SR-p53": {"learning_rate": 0.0001, "batch_size": 16}
}

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
        tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }
        
class MambaForSequenceClassificationFull(nn.Module):
    def __init__(self, base_model_name, num_labels, pos_weight):
        super(MambaForSequenceClassificationFull, self).__init__()
        # Load the pretrained model
        self.base_model = MambaLMHeadModel.from_pretrained(base_model_name).to(device)
        self.num_labels = num_labels
        self.pos_weight = torch.tensor([pos_weight]).to(device)  # Weight for positive class

        # Replace the lm_head with a classification head
        self.base_model.lm_head = nn.Sequential(
            nn.Dropout(0.1),  # Dropout regularization
            nn.Linear(self.base_model.config.d_model, num_labels)  # Classification head
        ).to(device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        
        # Forward pass through the backbone
        hidden_states = self.base_model.backbone(input_ids)  # Manually access the backbone
        #print("Hidden States Shape:", hidden_states.shape)  # Debugging output
        
        attention_mask = attention_mask.unsqueeze(-1)  # Shape: [batch_size, seq_length, 1]
        # Broadcast attention_mask to match hidden_states' shape
        attention_mask = attention_mask.expand_as(hidden_states)
    
        masked_hidden_states = hidden_states.masked_fill(attention_mask == 0, -float('inf'))
        pooled_output, _ = masked_hidden_states.max(dim=1)  # Max pooling across seq_length
      
        # Use the updated lm_head (classification head) for final logits
        final_logits = self.base_model.lm_head(pooled_output)  # Shape: [batch_size, num_labels]

        # Calculate loss 
        loss = None
        if labels is not None:
            labels = labels.float()  # Ensure labels are float for BCEWithLogitsLoss
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)  # Weighted BCE loss
            loss = loss_fn(final_logits.squeeze(-1), labels)

        return {"loss": loss, "logits": final_logits}
    
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
    
    
def train_and_evaluate(subtask, fold, run):
    print(f"Training {subtask}, Fold {fold}, Run {run}")
    train_df, val_df, test_df = load_crossval_data(subtask, fold)
    train_dataset = Tox21Dataset(train_df, tokenizer, subtask)
    val_dataset = Tox21Dataset(val_df, tokenizer, subtask)
    test_dataset = Tox21Dataset(test_df, tokenizer, subtask)

    hyperparams = best_hyperparams[subtask]
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    num_positives = len(train_dataset.data[train_dataset.data[subtask] == 1])
    num_negatives = len(train_dataset.data[train_dataset.data[subtask] == 0])
    pos_weight = num_negatives / num_positives 

    model = MambaForSequenceClassificationFull(base_model, num_labels=1, pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(20):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs['loss']
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                val_loss += outputs['loss'].item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} for fold {fold}, run {run}")
            break
        
    # Restore the best model's parameters before testing
    if best_model_state:
        model.load_state_dict(best_model_state)

    return evaluate_model(model, test_loader, subtask, fold, run)

def evaluate_model(model, test_loader, subtask, fold, run):
    model.eval()
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            all_logits.extend(outputs['logits'].cpu().numpy().flatten())
            all_labels.extend(batch['labels'].cpu().numpy())

    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    auc = roc_auc_score(all_labels, all_logits)

    print(f"Subtask: {subtask}, Fold: {fold}, Run: {run}")
    print(f"AUC-ROC: {auc:.4f}")

    return auc

def cross_validate(subtask):
    fold_aucs = []
    for fold in range(1, 6):
        run_aucs = []
        for run in range(1, 6):
            auc = train_and_evaluate(subtask, fold, run)
            run_aucs.append(auc)
        fold_mean_auc = np.mean(run_aucs)
        fold_aucs.append(fold_mean_auc)

    overall_mean_auc = np.mean(fold_aucs)
    overall_std_auc = np.std(fold_aucs)
    overall_se_auc = overall_std_auc / sqrt(len(fold_aucs))

    print(f"\nFinal Results for {subtask}:")
    print(f"Overall Mean AUC-ROC: {overall_mean_auc:.4f}, Std: {overall_std_auc:.4f}, SE: {overall_se_auc:.4f}")

for subtask in subtasks:
    cross_validate(subtask)