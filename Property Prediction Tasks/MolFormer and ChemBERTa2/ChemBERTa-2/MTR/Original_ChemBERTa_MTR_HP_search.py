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
from itertools import product
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(base_model)

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

# ChemBERTa model
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
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  
        logits = self.classification_head(cls_embeddings)

        loss = None
        if labels is not None:
            labels = labels.float().to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.squeeze(-1), labels)

        return {"loss": loss, "logits": logits}
    
# Hyperparameter search function
def crossval_hyperparameter_search(num_folds=5):
    learning_rates = [1e-4, 1e-5, 5e-5]
    batch_sizes = [8, 16]
    
    best_val_auc = -1.0
    best_config = None
    
    for lr, batch_size in product(learning_rates, batch_sizes):
        print(f"Testing config: LR={lr}, Batch Size={batch_size}")
        fold_aucs = []

        for fold in range(1, num_folds + 1):
            train_df, val_df, _ = load_fold_data(fold)
            train_dataset = PropertyDataset(train_df, tokenizer, target_column='Permeability_Label')
            val_dataset = PropertyDataset(val_df, tokenizer, target_column='Permeability_Label')

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
            
            model = ChemBERTaForSequenceClassification(base_model, num_labels=1)
            optimizer = AdamW(model.parameters(), lr=lr)
            
            for epoch in range(3):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                    loss = outputs['loss']
                    loss.backward()
                    optimizer.step()

            # Validation
            model.eval()
            all_labels = []
            all_logits = []
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                    all_logits.extend(outputs['logits'].cpu().numpy().flatten())
                    all_labels.extend(batch['labels'].cpu().numpy())
            
            all_labels = np.array(all_labels)
            all_logits = np.array(all_logits)
            auc = roc_auc_score(all_labels, all_logits)
            fold_aucs.append(auc)

        avg_auc = np.mean(fold_aucs)
        print(f"Config LR={lr}, Batch Size={batch_size}, Avg AUC={avg_auc:.4f}")

        if avg_auc > best_val_auc:
            best_val_auc = avg_auc
            best_config = {'learning_rate': lr, 'batch_size': batch_size}

    return best_val_auc, best_config

if __name__ == "__main__":
    best_val_auc, best_config = crossval_hyperparameter_search()
    print("Best Validation AUC:", best_val_auc)
    print("Best Hyperparameter Configuration:", best_config)