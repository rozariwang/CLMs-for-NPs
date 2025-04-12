# Data Loading and Preprocessing
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# Training
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import wandb
from sam import SAM

# Mamba and Tokenizer 
from tqdm import tqdm
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import json
from tokenisers import NPBPETokenizer

# Saving 
import logging
import struct
import os

wandb.login(key="your_wandb_api_key")  # Replace with your actual Wandb API key 

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

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, patience=5):
    best_val_loss = float('inf')  # Initialize best validation loss to a large value
    patience_counter = 0  # Tracks the number of epochs since validation loss improved
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
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
                
                # Log training loss every 10 batches 
                if batch_idx % 10 == 0:
                    avg_batch_loss = epoch_loss / (batch_idx + 1)
                    wandb.log({"train_loss_batch": avg_batch_loss, "epoch": epoch + 1, "batch": batch_idx + 1})

        avg_train_loss = epoch_loss / num_batches
        train_perplexity = torch.exp(torch.tensor(avg_train_loss))

        # Evaluate on validation set
        avg_val_loss, val_perplexity = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")
        
        # Log training and validation statistics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_perplexity": train_perplexity.item(),
            "val_loss": avg_val_loss,
            "val_perplexity": val_perplexity.item()
        })
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss  # Update best validation loss
            patience_counter = 0  # Reset patience counter
            torch.save(model.state_dict(), "M2_npbpe100_sfs_best_model.pth")  # Save the best model weights
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")

        # Check if patience has been exceeded
        if patience_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            model.load_state_dict(torch.load("M2_npbpe100_sfs_best_model.pth"))  # Load best model
            print("Best model restored for testing and saving.")
            break   
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##  Paths to files ##
train_data_file = "./hhwang/shiawaseda/Datasets/train_sf.txt"
val_data_file = "./hhwang/shiawaseda/Datasets/val_sf.txt"
test_data_file = "./hhwang/shiawaseda/Datasets/test_sf.txt"

## Mamba-2 ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = MambaConfig(
    d_model=512,
    n_layer=12,
    d_intermediate=2048, 
    vocab_size=100,
    ssm_cfg={'layer': 'Mamba2'},
    attn_layer_idx=[],
    attn_cfg={},
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True
)

model = MambaLMHeadModel(config)
model = model.to(device)

base_optimizer = optim.Adam
# Initialize SAM optimizer
optimizer = SAM(model.parameters(), base_optimizer, lr=0.0005, rho=0.05, weight_decay=0.0001)
criterion = CrossEntropyLoss(ignore_index=0)

tokenizer = NPBPETokenizer("./hhwang/shiawaseda/Datasets/npbpe_100.json")
# Preprocess the datasets
train_dataset = SMILESDataset(train_data_file, tokenizer, max_length=512)
val_dataset = SMILESDataset(val_data_file, tokenizer, max_length=512)
test_dataset = SMILESDataset(test_data_file, tokenizer, max_length=512)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

num_epochs = 150

wandb.init(project="mamba-2-npbpe100-sfs-training", config={
    "learning_rate": 0.0005,
    "epochs": num_epochs,
    "batch_size": train_loader.batch_size,
    "d_model": model.config.d_model,
    "n_layer": model.config.n_layer
})

train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, patience=5)

# Testing using the evaluate function
test_loss, test_perplexity = evaluate(model, test_loader, criterion, device)
wandb.log({"test_loss": test_loss, "test_perplexity": test_perplexity})
print(f'Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.4f}')
        
def save_pretrained(model, tokenizer, save_directory):
    os.makedirs(save_directory, exist_ok=True)

    model_path = os.path.join(save_directory, 'pytorch_model.bin')
    torch.save(model.state_dict(), model_path)

    config_path = os.path.join(save_directory, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(model.config.__dict__, f, indent=4)

    vocab = tokenizer.tokenizer.get_vocab() if hasattr(tokenizer.tokenizer, "get_vocab") else {}
    vocab_path = os.path.join(save_directory, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=4)

    print(f"Model and tokenizer saved to {save_directory}.")


def dtype_to_abbreviation(dtype):
    # This function maps PyTorch dtypes to their abbreviations
    dtype_str = str(dtype)
    return {
        'torch.float32': 'F32',
        'torch.float64': 'F64',
        'torch.float16': 'F16',
        'torch.int32': 'I32',
        'torch.int64': 'I64',
        'torch.int16': 'I16',
        'torch.int8': 'I8',
        'torch.uint8': 'U8'
    }.get(dtype_str, dtype_str)  


def save_model_as_safetensors(model, save_directory, filename='model.safetensors'):
    model.to('cpu')  # Move the model to CPU to handle any potential device-specific tensors
    state_dict = model.state_dict()
    metadata = {'__metadata__': {'format': 'pt'}}
    tensor_data = bytearray()

    current_offset = 0
    for name, tensor in state_dict.items():
        tensor_bytes = tensor.numpy().tobytes()
        dtype_abbreviation = dtype_to_abbreviation(tensor.dtype)  # Convert dtype to abbreviation
        metadata[name] = {
            'dtype': dtype_abbreviation,  # Use abbreviated dtype
            'shape': list(tensor.shape),
            'data_offsets': [current_offset, current_offset + len(tensor_bytes)]
        }
        current_offset += len(tensor_bytes)
        tensor_data.extend(tensor_bytes)

    metadata_json = json.dumps(metadata)
    metadata_bytes = metadata_json.encode('utf-8')
    metadata_length = len(metadata_bytes)

    # Open the file in binary mode and write
    with open(os.path.join(save_directory, filename), 'wb') as f:
        f.write(struct.pack('<Q', metadata_length))  # Write the length of metadata as an unsigned long long
        f.write(metadata_bytes)  # Write the metadata
        f.write(tensor_data)  # Write the actual tensor data

    return os.path.join(save_directory, filename)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the save directory
save_directory = './Mamba2_npbpe100_scaffold_split'

try:
    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Save using the original save_pretrained method
    save_pretrained(model, tokenizer, save_directory)
    logging.info(f"Standard model files saved in {save_directory}.")

    # Save using custom safetensors method
    safetensors_path = save_model_as_safetensors(model, save_directory)
    logging.info(f"Model saved successfully in custom safetensors format at {safetensors_path}.")
    
    # Log the model files as WandB artifacts
    artifact = wandb.Artifact('mamba2-npbpe100-scaffold-split', type='model')

    # Add both saved files (pretrained and safetensors) to the artifact
    artifact.add_file(os.path.join(save_directory, 'pytorch_model.bin'))  
    artifact.add_file(safetensors_path)  # Safetensors file

    # Log the artifact to WandB
    wandb.log_artifact(artifact)
    logging.info(f"Model artifacts logged to WandB.")

except Exception as e:
    logging.error(f"An error occurred while saving the model: {str(e)}")

wandb.finish()