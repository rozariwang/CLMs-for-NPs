import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import torch
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch import nn
import pandas as pd
from huggingface_hub import login
from itertools import product
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import matthews_corrcoef
from transformers import AutoTokenizer
from tokenisers import NPBPETokenizer
from tokenisers import CharLevelTokenizer
from tokenisers import AISTokenizer

wandb.login(key="your_wandb_api_key")  # Replace with your actual Wandb API key   
login(token="your_huggingface_token")  # Replace with your actual Hugging Face API token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenizer mapping
TOKENIZER_CLASSES = {
    "Char": CharLevelTokenizer,
    "BPE": AutoTokenizer.from_pretrained,
    "AIS": AISTokenizer,
    "npbpe60": NPBPETokenizer,
    "npbpe100": NPBPETokenizer,
    "npbpe1000": NPBPETokenizer,
    "npbpe7924": NPBPETokenizer,
    "npbpe30k": NPBPETokenizer
}

TOKENIZER_PATHS = {
    "Char": './hhwang/shiawaseda/Datasets/vocab.json',
    "BPE": "seyonec/PubChem10M_SMILES_BPE_450k",
    "AIS": './hhwang/shiawaseda/Datasets/ais_vocab.json',
    "npbpe60": './hhwang/shiawaseda/Datasets/npbpe_60.json',
    "npbpe100": './hhwang/shiawaseda/Datasets/npbpe_100.json',
    "npbpe1000": './hhwang/shiawaseda/Datasets/npbpe_1000.json',
    "npbpe7924": './hhwang/shiawaseda/Datasets/npbpe_7924vocab.json',
    "npbpe30k": './hhwang/shiawaseda/Datasets/npbpe_tokenizer.json'
}

# Iterate through models and tokenizers
model_names = [
    "rozariwang/M2-Char-sfs", "rozariwang/M1-Char-sfs", "rozariwang/GPT-Char-sfs",
    "rozariwang/M2-BPE-sfs", "rozariwang/M1-BPE-sfs", "rozariwang/GPT-BPE-sfs",
    "rozariwang/M2-AIS-sfs", "rozariwang/M1-AIS-sfs", "rozariwang/GPT-AIS-sfs",
    "rozariwang/M2-npbpe60-sfs", "rozariwang/M1-npbpe60-sfs", "rozariwang/GPT-npbpe60-sfs",
    "rozariwang/M2-npbpe100-sfs", "rozariwang/M1-npbpe100-sfs", "rozariwang/GPT-npbpe100-sfs",
    "rozariwang/M2-npbpe1000-sfs", "rozariwang/M1-npbpe1000-sfs", "rozariwang/GPT-npbpe1000-sfs",
    "rozariwang/M2-npbpe7924-sfs", "rozariwang/M1-npbpe7924-sfs", "rozariwang/GPT-npbpe7924-sfs",
    "rozariwang/M2-npbpe30k-sfs", "rozariwang/M1-npbpe30k-sfs", "rozariwang/GPT-npbpe30k-sfs"
]
        
class AntiCancerDataset(Dataset):
    def __init__(self, data, tokenizer, target_column, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.target_column = target_column
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Standardized_SMILES']
        label = self.data.iloc[idx][self.target_column]
        tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Model classes
class MambaForSequenceClassificationFull(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super(MambaForSequenceClassificationFull, self).__init__()
        self.base_model = MambaLMHeadModel.from_pretrained(base_model_name).to(device)
        self.num_labels = num_labels
        self.base_model.lm_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.base_model.config.d_model, num_labels)
        ).to(device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        hidden_states = self.base_model.backbone(input_ids)
        attention_mask = attention_mask.unsqueeze(-1)
        attention_mask = attention_mask.expand_as(hidden_states)
        masked_hidden_states = hidden_states.masked_fill(attention_mask == 0, -float('inf'))
        pooled_output, _ = masked_hidden_states.max(dim=1)
        final_logits = self.base_model.lm_head(pooled_output)
        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(final_logits.squeeze(-1), labels)
        return {"loss": loss, "logits": final_logits}

class GPT2ForSequenceClassification(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super(GPT2ForSequenceClassification, self).__init__()
        self.base_model = GPT2Model.from_pretrained(base_model_name).to(device)
        self.num_labels = num_labels
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.base_model.config.hidden_size, num_labels)
        ).to(device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  
        attention_mask = attention_mask.unsqueeze(-1)  # Shape: [batch_size, seq_length, 1]
        attention_mask = attention_mask.expand_as(hidden_states)
        masked_hidden_states = hidden_states.masked_fill(attention_mask == 0, -float('inf'))
        pooled_output, _ = masked_hidden_states.max(dim=1)  # Max pooling across seq_length
        logits = self.classification_head(pooled_output)  # Shape: [batch_size, num_labels]

        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.squeeze(-1), labels)

        return {"loss": loss, "logits": logits}

# Collate function
def collate_batch(batch, tokenizer_type, tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    if tokenizer_type == "Char" or tokenizer_type == "AIS":
        padding_value = tokenizer.vocab['[PAD]']
    elif tokenizer_type == "BPE":
        padding_value = tokenizer.pad_token_id
    elif "npbpe" in tokenizer_type:
        padding_value = tokenizer.tokenizer.token_to_id('[PAD]')
    
    batch_padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=padding_value).to(device)
    attention_mask = (batch_padded_inputs != padding_value).float().to(device)
    labels = torch.tensor(labels, dtype=torch.float).to(device)
    
    return {'input_ids': batch_padded_inputs, 'attention_mask': attention_mask, 'labels': labels}


# Hyperparameter search function
def crossval_hyperparameter_search(model_name, tokenizer, num_folds=5):
    learning_rates = [1e-4, 1e-5, 5e-5]
    batch_sizes = [8, 16]
    best_val_mcc, best_config = -1.0, None

    for lr, batch_size in product(learning_rates, batch_sizes):
        print(f"Testing config: LR={lr}, Batch Size={batch_size} for model {model_name}")
        fold_mccs = []
        target_column = 'Activity_Label'

        for fold in range(1, num_folds + 1):
            train_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Anti_Cancer_CV/cancer_activity_fold{fold}_rd_train.csv')
            val_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Anti_Cancer_CV/cancer_activity_fold{fold}_rd_val.csv')
            
            train_dataset = AntiCancerDataset(train_df, tokenizer, target_column)
            val_dataset = AntiCancerDataset(val_df, tokenizer, target_column)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=lambda batch: collate_batch(batch, model_name.split('-')[1], tokenizer)
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch: collate_batch(batch, model_name.split('-')[1], tokenizer)
            )
            
            model = MambaForSequenceClassificationFull(model_name, num_labels=1) if 'M' in model_name else GPT2ForSequenceClassification(model_name, num_labels=1)
            optimizer = AdamW(model.parameters(), lr=lr)
            
            for epoch in range(3):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                    loss = outputs['loss']
                    loss.backward()
                    optimizer.step()
            
            all_preds, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                    #preds = torch.argmax(outputs['logits'], dim=1).cpu().numpy()
                    logits = outputs['logits']
                    if logits.shape[-1] == 1:
                        logits = logits.squeeze(-1)
                    preds = torch.sigmoid(logits).round().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch['labels'].float().cpu().numpy())
            mcc = matthews_corrcoef(all_labels, all_preds)
            fold_mccs.append(mcc)
        
        avg_mcc = np.mean(fold_mccs)
        #print(f"Config LR={lr}, Batch Size={batch_size}, Avg MCC={avg_mcc:.4f} for model {model_name}")
        if avg_mcc > best_val_mcc:
            best_val_mcc, best_config = avg_mcc, {'learning_rate': lr, 'batch_size': batch_size}
    return best_val_mcc, best_config


# Fine-tuning function
def train_and_evaluate(model_name, tokenizer, best_config, fold, run):
    print(f"Starting Fine-Tuning for {model_name}, Fold {fold}, Run {run}")
    train_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Anti_Cancer_CV/cancer_activity_fold{fold}_rd_train.csv')
    val_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Anti_Cancer_CV/cancer_activity_fold{fold}_rd_val.csv')
    test_df = pd.read_csv(f'./hhwang/shiawaseda/Downstream_Data/Anti_Cancer_CV/cancer_activity_fold{fold}_rd_test.csv')
    target_column = 'Activity_Label'
    
    train_dataset = AntiCancerDataset(train_df, tokenizer, target_column)
    val_dataset = AntiCancerDataset(val_df, tokenizer, target_column)
    test_dataset = AntiCancerDataset(test_df, tokenizer, target_column)
    
    batch_size = best_config['batch_size']
    learning_rate = best_config['learning_rate']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, model_name.split('-')[1], tokenizer)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, model_name.split('-')[1], tokenizer)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, model_name.split('-')[1], tokenizer)
    )
    
    model = GPT2ForSequenceClassification(model_name, num_labels=1) if 'GPT' in model_name else MambaForSequenceClassificationFull(model_name, num_labels=1)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0
    best_model_state = None
    
    for epoch in range(25):  # Early stopping with max epochs
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

    # Load the best model state
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            logits = outputs['logits']
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            preds = torch.sigmoid(logits).round().cpu().numpy()
            all_logits.extend(preds)
            all_labels.extend(batch['labels'].float().cpu().numpy()) 

    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    mcc = matthews_corrcoef(all_labels, all_logits)

    print(f"Model: {model_name}, Fold: {fold}, MCC: {mcc:.4f}")
    return mcc

# Cross-validation and repeated runs
def cross_validate(model_name, tokenizer, best_config):
    fold_mccs = []
    for fold in range(1, 6):
        run_mccs = []
        for run in range(1, 6):
            mcc = train_and_evaluate(model_name, tokenizer, best_config, fold, run)
            run_mccs.append(mcc)
        fold_mean_mcc = np.mean(run_mccs)
        fold_mccs.append(fold_mean_mcc)

    overall_mean_mcc = np.mean(fold_mccs)
    overall_std_mcc = np.std(fold_mccs)
    overall_se_mcc = overall_std_mcc / np.sqrt(len(fold_mccs))

    print(f"\nFinal Results for {model_name}:")
    print(f"Overall Mean MCC: {overall_mean_mcc:.4f}, Std: {overall_std_mcc:.4f}, SE: {overall_se_mcc:.4f}")

for model_name in model_names:
    tokenizer_type = model_name.split('-')[1]  # Extract tokenizer type from model name
    tokenizer_path = TOKENIZER_PATHS[tokenizer_type]
    tokenizer_class = TOKENIZER_CLASSES[tokenizer_type]
    tokenizer = tokenizer_class(tokenizer_path) if tokenizer_type != "BPE" else AutoTokenizer.from_pretrained(tokenizer_path)

    best_val_mcc, best_config = crossval_hyperparameter_search(model_name, tokenizer)
    #print(f"Best Validation MCC for {model_name}: {best_val_mcc}")
    print(f"Best Hyperparameter Configuration for {model_name}: {best_config}")
    cross_validate(model_name, tokenizer, best_config)