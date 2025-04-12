import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import numpy as np
import csv
import os
from sklearn.metrics import roc_auc_score


# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = "DeepChem/ChemBERTa-77M-MLM"
model_path = "./ChemBERTa2_finetuned_model.pth"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load fold data function
def load_fold_data(fold):
    train_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Peptides_CV/peptides_fold{fold}_sf_train.csv')
    val_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Peptides_CV/peptides_fold{fold}_sf_val.csv')
    test_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Peptides_CV/peptides_fold{fold}_sf_test.csv')
    return train_df, val_df, test_df

class PropertyDataset(Dataset):
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

# Collate function
def collate_batch(batch):
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

# ChemBERTa model for binary or multi-label classification
class ChemBERTaForSequenceClassification(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super(ChemBERTaForSequenceClassification, self).__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name).to(device)
        self.num_labels = num_labels
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.base_model.config.hidden_size, num_labels)
        ).to(device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
        logits = self.classification_head(cls_embeddings)

        loss = None
        if labels is not None:
            labels = labels.float().to(device) 
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.squeeze(-1), labels)

        return {"loss": loss, "logits": logits}

# Training and evaluation function
def train_and_evaluate(fold, run):
    print(f"Starting Fold {fold}, Run {run}")
    train_df, val_df, test_df = load_fold_data(fold)
    target_column = 'Permeability_Label'  

    train_dataset = PropertyDataset(train_df, tokenizer, target_column)
    val_dataset = PropertyDataset(val_df, tokenizer, target_column)
    test_dataset = PropertyDataset(test_df, tokenizer, target_column)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)

    model = ChemBERTaForSequenceClassification(base_model, num_labels=1)
    #model.load_state_dict(torch.load(model_path, map_location=device))  # Load fine-tuned weights
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {key.replace("roberta", "base_model"): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    optimizer = AdamW(model.parameters(), lr=0.0001)

    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(50):
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

    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            logits = outputs['logits'].cpu().numpy().flatten()
            labels = batch['labels'].cpu().numpy().flatten()
            
            all_labels.extend(labels)
            all_logits.extend(logits)

    auc = roc_auc_score(np.array(all_labels), np.array(all_logits))
    print(f"Fold: {fold}, Run: {run}, AUC: {auc:.4f}")
    return auc

# Cross-validation
def cross_validate():
    fold_aucs = []
    for fold in range(1, 6):
        run_aucs = []
        for run in range(1, 6):
            auc = train_and_evaluate(fold, run)
            run_aucs.append(auc)
        fold_mean_auc = np.mean(run_aucs)
        fold_aucs.append(fold_mean_auc)

    overall_mean_auc = np.mean(fold_aucs)
    overall_std_auc = np.std(fold_aucs)
    overall_se_auc = overall_std_auc / np.sqrt(len(fold_aucs))
    print(f"\nFinal Results:")
    print(f"Overall Mean AUC: {overall_mean_auc:.4f}, Std: {overall_std_auc:.4f}, SE: {overall_se_auc:.4f}")

if __name__ == "__main__":
    cross_validate()