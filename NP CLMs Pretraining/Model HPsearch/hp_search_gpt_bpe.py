import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
import pandas as pd
import random
from sam import SAM
from tqdm import tqdm

# Data Loading and Preprocessing
class SMILESDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        super(SMILESDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.smiles = []

        # Load the data file and process each line
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:  # Ensure the line is not empty
                    self.smiles.append(line)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_string = self.smiles[idx]
        tokenized = self.tokenizer.encode(smiles_string, add_special_tokens=True, max_length=self.max_length, truncation=True)
        tensor = torch.tensor(tokenized, dtype=torch.long)
        return tensor


def collate_batch(batch):
    padding_value = tokenizer.pad_token_id
    batch_padded = pad_sequence(batch, batch_first=True, padding_value=padding_value)
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

# Training Function 
def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, hyperparams):
    print(f"Training with hyperparameters: n_embd={hyperparams['n_embd']}, n_layer={hyperparams['n_layer']}, n_head={hyperparams['n_head']}, lr={hyperparams['lr']}")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
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

                pbar.update(1)  # Update progress bar

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

tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

# Load the datasets
total_train_dataset = SMILESDataset(train_data_file, tokenizer, max_length=512)
total_val_dataset = SMILESDataset(val_data_file, tokenizer, max_length=512)

print(f"Total training samples: {len(total_train_dataset)}")
print(f"Total validation samples: {len(total_val_dataset)}")

# Select 5% of the training and validation data randomly
train_subset_indices = torch.randperm(len(total_train_dataset))[:int(0.05 * len(total_train_dataset))]
print(f"Train subset size: {len(train_subset_indices)}")
train_subset_dataset = Subset(total_train_dataset, train_subset_indices)
val_subset_indices = torch.randperm(len(total_val_dataset))[:int(0.05 * len(total_val_dataset))]
val_subset_dataset = Subset(total_val_dataset, val_subset_indices)

print(f"Length of train_subset_dataset: {len(train_subset_dataset)}")
print(f"Length of val_subset_dataset: {len(val_subset_dataset)}")

# Create DataLoader 
train_loader = DataLoader(train_subset_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_subset_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

# Define the range for each parameter
n_embd_options = [256, 512]
n_layer_options = [4, 8, 12]
n_head_options = [2, 4, 8]
lr_options = [0.001, 0.005, 0.0001, 0.0005]
num_epochs = 5
num_search_iters = 36  

results = []

# Set to store tested hyperparameter combinations
tested_combinations = set()

# Random search
for _ in range(num_search_iters):
    # Generate a unique combination of hyperparameters
    while True:
        n_embd = random.choice(n_embd_options)
        n_layer = random.choice(n_layer_options)
        n_head = random.choice(n_head_options)
        lr = random.choice(lr_options)
        combination = (n_embd, n_layer, n_head, lr)

        # Check if this combination has already been tested
        if combination not in tested_combinations:
            tested_combinations.add(combination)
            break

    config = GPT2Config(
        vocab_size=len(tokenizer),  
        n_positions=512,  # Maximum sequence length
        n_embd=n_embd,      
        n_layer=n_layer,  # Number of Transformer layers
        n_head=n_head,       
        n_inner=n_embd * 4,     
        bos_token_id=0,  
        eos_token_id=2   
    )
    model = GPT2LMHeadModel(config)
    model = model.to(device)
   
    
    base_optimizer = optim.Adam
    # Initialize SAM optimizer
    optimizer = SAM(model.parameters(), base_optimizer, lr=lr, rho=0.05, weight_decay=0.0001)
    criterion = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Hyperparameters to pass to train function
    hyperparams = {
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'lr': lr
    }

    # training
    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, hyperparams)
    # Evaluation
    avg_val_loss, val_perplexity = evaluate(model, val_loader, criterion, device)

    # Store results
    results.append({
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'lr': lr,
        'val_loss': avg_val_loss
    })

# Save results to DataFrame and then to CSV file
results_df = pd.DataFrame(results)
# Print results
print("Random search results summary:")
print(results_df)
best_result = results_df.loc[results_df['val_loss'].idxmin()]
print("\nBest hyperparameter set for GPT_BPE_sfs:")
print(best_result.to_dict())

