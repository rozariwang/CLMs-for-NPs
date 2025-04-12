import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import numpy as np
from itertools import product

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = "ibm/MoLFormer-XL-both-10pct"
model_path = "./MoLFormer_finetuned_model.pth"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

def load_fold_data(fold):
    train_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Anti_Cancer_CV/cancer_activity_fold{fold}_sf_train.csv')
    val_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Anti_Cancer_CV/cancer_activity_fold{fold}_sf_val.csv')
    test_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Anti_Cancer_CV/cancer_activity_fold{fold}_sf_test.csv')
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

# MolFormer model
class MolFormerForSequenceClassification(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super(MolFormerForSequenceClassification, self).__init__()
        self.base_model = AutoModel.from_pretrained(
            base_model_name, num_labels=num_labels, deterministic_eval=True, trust_remote_code=True
        ).to(device)
        self.num_labels = num_labels
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.base_model.config.hidden_size, num_labels)
        ).to(device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classification_head(outputs.last_hidden_state[:, 0, :])  # [CLS] token representation

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
    
    best_val_mcc = -1.0
    best_config = None
    
    for lr, batch_size in product(learning_rates, batch_sizes):
        print(f"Testing config: LR={lr}, Batch Size={batch_size}")
        fold_mccs = []

        for fold in range(1, num_folds + 1):
            train_df, val_df, _ = load_fold_data(fold)
            train_dataset = PropertyDataset(train_df, tokenizer, target_column='Activity_Label')
            val_dataset = PropertyDataset(val_df, tokenizer, target_column='Activity_Label')

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
            
            model = MolFormerForSequenceClassification(base_model, num_labels=1)
            #model.load_state_dict(torch.load(model_path, map_location=device))  # Load fine-tuned weights
            state_dict = torch.load(model_path, map_location=device)
            # Rename the keys from "molformer.*" to "base_model.molformer.*"
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key if key.startswith("base_model.") else "base_model." + key
                new_state_dict[new_key] = value
            # Load the modified state dictionary
            model.load_state_dict(new_state_dict, strict=False)  # `strict=False` allows partial loading
            #for name, param in model.named_parameters():
                #print(f"{name}: {param.mean().item()}")

            optimizer = AdamW(model.parameters(), lr=lr)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for batch in val_loader:
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                    logits = outputs['logits']  # Raw scores

                    # Convert logits to probabilities
                    probs = torch.sigmoid(logits).cpu().numpy()  

                    # Convert probabilities to binary predictions
                    preds = (probs > 0.5).astype(int)  

                    # Store predictions and true labels
                    all_preds.extend(preds.flatten())  
                    all_labels.extend(batch['labels'].cpu().numpy().flatten())

            mcc = matthews_corrcoef(all_labels, all_preds)
            fold_mccs.append(mcc)

        avg_mcc = np.mean(fold_mccs)
        print(f"Config LR={lr}, Batch Size={batch_size}, Avg MCC={avg_mcc:.4f}")

        if avg_mcc > best_val_mcc:
            best_val_mcc = avg_mcc
            best_config = {'learning_rate': lr, 'batch_size': batch_size}

    return best_val_mcc, best_config

if __name__ == "__main__":
    best_val_mcc, best_config = crossval_hyperparameter_search()
    print("Best Validation MCC:", best_val_mcc)
    print("Best Hyperparameter Configuration:", best_config)