import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
# Data Loading and Preprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
# Training
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from sam import SAM
# Mamba and Tokenizer 
from tokenisers import CharLevelTokenizer
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import pandas as pd
import random


# Data Loading and Preprocessing
class SMILESDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        super(SMILESDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.smiles = []

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:  
                    self.smiles.append(line)

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles_string = self.smiles[idx]
        tokenized = self.tokenizer.encode(smiles_string, add_special_tokens=True, max_length=self.max_length, truncation=True)
        tensor = torch.tensor(tokenized, dtype=torch.long)
        return tensor

def collate_batch(batch):
    batch_padded = pad_sequence(batch, batch_first=True, padding_value=0)
    inputs = batch_padded[:, :-1]
    targets = batch_padded[:, 1:] 
    inputs = inputs.long()
    return inputs, targets

# Evaluation Function 
def evaluate(model, data_loader, criterion, device):
    model.eval()
    eval_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1).long()

            # Forward pass
            outputs = model(inputs)
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1)

            loss = criterion(logits, targets)
            eval_loss += loss.item()
            num_batches += 1

    avg_loss = eval_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, hyperparams):
    print(f"Training with hyperparameters: d_model={hyperparams['d_model']}, n_layer={hyperparams['n_layer']}, lr={hyperparams['lr']}")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for _, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device).long()
            targets = targets.to(device).view(-1).long()

            def closure():
                optimizer.zero_grad()  # Reset gradients
                outputs = model(inputs.long())  # Forward pass
                logits = outputs.logits  # Access logits
                logits = logits.view(-1, logits.size(-1))  # Reshape logits for loss calculation
                targets_res = targets.view(-1)  # Reshape targets
                loss = criterion(logits, targets_res)  # Calculate loss
                loss.backward()  # Backward pass (calculate gradients)
                return loss

            # Perform the SAM optimizer step
            loss = closure()  # First, compute the loss and gradients
            optimizer.step(closure)  # Then, perform the optimizer step with SAM

            # Update epoch loss
            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_perplexity = torch.exp(torch.tensor(avg_train_loss))

        # Evaluate on validation set
        avg_val_loss, val_perplexity = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")
   
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_file = "./hhwang/shiawaseda/Datasets/train_sf.txt"
val_data_file = "./hhwang/shiawaseda/Datasets/val_sf.txt"

tokenizer = CharLevelTokenizer(vocab_file="./hhwang/shiawaseda/Datasets/vocab.json")

# Load datasets
total_train_dataset = SMILESDataset(train_data_file, tokenizer, max_length=512)
total_val_dataset = SMILESDataset(val_data_file, tokenizer, max_length=512)
# Select 5% of the training and validation data randomly
train_subset_indices = torch.randperm(len(total_train_dataset))[:int(0.05 * len(total_train_dataset))]
train_subset_dataset = Subset(total_train_dataset, train_subset_indices)
val_subset_indices = torch.randperm(len(total_val_dataset))[:int(0.05 * len(total_val_dataset))]
val_subset_dataset = Subset(total_val_dataset, val_subset_indices)
# Create DataLoader 
train_loader = DataLoader(train_subset_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_subset_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

# Define the range for each parameter
d_model_options = [256, 512]
n_layer_options = [4, 8, 12]
lr_options = [0.001, 0.005, 0.0001, 0.0005]
num_epochs = 5
num_search_iters = 12  

results = []

# Set to store tested hyperparameter combinations
tested_combinations = set()

# Random search
for _ in range(num_search_iters):
    # Generate a unique combination of hyperparameters
    while True:
        d_model = random.choice(d_model_options)
        n_layer = random.choice(n_layer_options)
        lr = random.choice(lr_options)
        combination = (d_model, n_layer, lr)

        # Check if this combination has already been tested
        if combination not in tested_combinations:
            tested_combinations.add(combination)
            break

    config = MambaConfig(
        d_model=d_model,
        n_layer=n_layer,
        d_intermediate=d_model * 4,
        vocab_size=48, 
        ssm_cfg={'layer': 'Mamba1'},
        attn_layer_idx=[],
        attn_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True
    )
    model = MambaLMHeadModel(config).to(device)
    
    base_optimizer = optim.Adam
    # Initialize SAM optimizer
    optimizer = SAM(model.parameters(), base_optimizer, lr=lr, rho=0.05, weight_decay=0.0001)
    criterion = CrossEntropyLoss(ignore_index=0)
    
    # Hyperparameters to pass to train function
    hyperparams = {
        'd_model': d_model,
        'n_layer': n_layer,
        'lr': lr
    }

    # Use the train function
    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, hyperparams)
    # Evaluate the model after training
    avg_val_loss, val_perplexity = evaluate(model, val_loader, criterion, device)

    results.append({
        'd_model': d_model,
        'n_layer': n_layer,
        'lr': lr,
        'val_loss': avg_val_loss
    })

results_df = pd.DataFrame(results)

# Print results
print("Random search results summary:")
print(results_df)
best_result = results_df.loc[results_df['val_loss'].idxmin()]
print("\nBest hyperparameter set for M1_Char_sf:")
print(best_result.to_dict())