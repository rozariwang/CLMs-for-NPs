import os
import torch
import wandb
import json
import torch.optim as optim
from transformers import GPT2Config, GPT2LMHeadModel
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sam import SAM
from tokenisers import AISTokenizer, CharLevelTokenizer, NPBPETokenizer
from transformers import AutoTokenizer
from tqdm import tqdm
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
# Saving 
import logging
import struct

class SMILESDataset(Dataset):
    """
    PyTorch Dataset for loading and tokenizing SMILES strings from a text file.

    Args:
        file_path (str): Path to the SMILES text file (one SMILES string per line).
        tokenizer (Tokenizer): Tokenizer to convert SMILES strings into token IDs.
        max_length (int): Maximum token sequence length.

    Returns:
        torch.Tensor: Tokenized SMILES string as a tensor.
    """
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

# Collate function
def collate_batch(batch, tokenizer=None):
    """
    Collate function for padding a batch of tokenized SMILES tensors.

    Args:
        batch (List[torch.Tensor]): List of tokenized SMILES sequences.
        tokenizer (Tokenizer, optional): Tokenizer to retrieve the padding token ID.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of (input sequences, target sequences).
    """
    padding_value = tokenizer.pad_token_id if tokenizer and hasattr(tokenizer, 'pad_token_id') else 0
    batch_padded = pad_sequence(batch, batch_first=True, padding_value=padding_value)
    inputs = batch_padded[:, :-1]
    targets = batch_padded[:, 1:]  
    inputs = inputs.long()
    return inputs, targets

# Evaluation Function 
def evaluate(model, data_loader, criterion, device):
    """
    Evaluates a model using cross-entropy loss and computes perplexity.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for validation or test data.
        criterion (Loss): Loss function (CrossEntropyLoss).
        device (torch.device): Device to run evaluation on.

    Returns:
        Tuple[float, torch.Tensor]: Average loss and perplexity.
    """
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

# Token ID retrieval
def get_token_id(tokenizer, token, tokenizer_type):
    """
    Retrieves the token ID for a specific token across different tokenizer types.

    Args:
        tokenizer (Tokenizer): The tokenizer used.
        token (str): The token whose ID is to be retrieved.
        tokenizer_type (str): Type of tokenizer (e.g., 'bpe', 'ais', 'npbpe').

    Returns:
        int: Token ID corresponding to the input token.

    Raises:
        ValueError: If tokenizer type is not supported.
    """
    if tokenizer_type.startswith("npbpe"):
        return tokenizer.tokenizer.token_to_id(token)
    elif tokenizer_type in ["ais", "char"]:
        return tokenizer.vocab[token]
    elif tokenizer_type == "bpe":
        return tokenizer.convert_tokens_to_ids(token)
    else:
        raise ValueError(f"Unsupported tokenizer type for token ID lookup: {tokenizer_type}")


def save_pretrained(model, tokenizer, save_directory):
    """
    Saves a model and tokenizer in regular and safetensors formats.

    Args:
        model (torch.nn.Module): The trained model.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        save_directory (str): Directory to save the model and tokenizer files.

    Files Saved:
        - pytorch_model.bin: Model weights
        - config.json: Model configuration
        - vocab.json: Tokenizer vocabulary
    """
    os.makedirs(save_directory, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))

    # Save model config
    with open(os.path.join(save_directory, 'config.json'), 'w') as f:
        json.dump(model.config.__dict__, f, indent=4)

    # Retrieve and sanitize vocab
    vocab = {}
    try:
        if hasattr(tokenizer, 'vocab'):
            vocab = tokenizer.vocab
        elif hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
        elif hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'get_vocab'):
            vocab = tokenizer.tokenizer.get_vocab()

        # Ensure vocab is JSON-serializable
        vocab = {str(k): int(v) for k, v in vocab.items()}
    except Exception as e:
        logging.warning(f"Could not serialize tokenizer vocab: {e}")
        vocab = {}

    # Save vocab
    with open(os.path.join(save_directory, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=4)

    print(f"Model and tokenizer saved to {save_directory}.")


def dtype_to_abbreviation(dtype):
    return {
        'torch.float32': 'F32', 'torch.float64': 'F64', 'torch.float16': 'F16',
        'torch.int32': 'I32', 'torch.int64': 'I64', 'torch.int16': 'I16',
        'torch.int8': 'I8', 'torch.uint8': 'U8'
    }.get(str(dtype), str(dtype))

def save_model_as_safetensors(model, save_directory, filename='model.safetensors'):
    """
    Saves model weights in a custom safetensors-compatible binary format.

    Args:
        model (torch.nn.Module): Trained model to be saved.
        save_directory (str): Directory to save the safetensor file.
        filename (str): Name of the safetensor file.

    Returns:
        str: Full path to the saved safetensors file.
    """
    model.to('cpu')
    state_dict = model.state_dict()
    metadata = {'__metadata__': {'format': 'pt'}}
    tensor_data = bytearray()
    current_offset = 0
    for name, tensor in state_dict.items():
        tensor_bytes = tensor.numpy().tobytes()
        dtype_abbreviation = dtype_to_abbreviation(tensor.dtype)
        metadata[name] = {
            'dtype': dtype_abbreviation,
            'shape': list(tensor.shape),
            'data_offsets': [current_offset, current_offset + len(tensor_bytes)]
        }
        current_offset += len(tensor_bytes)
        tensor_data.extend(tensor_bytes)
    metadata_json = json.dumps(metadata).encode('utf-8')
    with open(os.path.join(save_directory, filename), 'wb') as f:
        f.write(struct.pack('<Q', len(metadata_json)))
        f.write(metadata_json)
        f.write(tensor_data)
    return os.path.join(save_directory, filename)


# Main entry function
def run_pretraining(config): 
    """
    Executes the full pretraining loop for a selected model and tokenizer.

    Args:
        config (dict): Configuration dictionary with model and training parameters.

    Process Overview:
        - Loads datasets and tokenizer
        - Initializes model and optimizer
        - Trains model with SAM optimizer
        - Applies early stopping
        - Evaluates on test data
        - Saves model in multiple formats and logs artifacts to Weights & Biases
    """  
    model_type = config['model'].lower()
    tokenizer_type = config['tokenizer'].lower()
    split_type = config['split'].lower()
    n_embd = config['n_embd']
    n_layer = config['n_layer']
    lr = config['lr']
    n_head = config.get('n_head')
    max_epochs = 150

    vocab_map = {
        'ais': 'ais_vocab.json', 'char': 'vocab.json', 'bpe': 'seyonec/PubChem10M_SMILES_BPE_450k',
        'npbpe60': 'npbpe_60.json', 'npbpe100': 'npbpe_100.json', 'npbpe1000': 'npbpe_1000.json',
        'npbpe7924': 'npbpe_7924vocab.json', 'npbpe30k': 'npbpe_tokenizer.json'
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data", "1M_NPs")
    vocab_dir = os.path.join(script_dir, "vocab_files")
    train_file = os.path.join(data_dir, f"train_{'sf' if split_type == 'scaffold' else 'rd'}.txt")
    val_file = os.path.join(data_dir, f"val_{'sf' if split_type == 'scaffold' else 'rd'}.txt")
    test_file = os.path.join(data_dir, f"test_{'sf' if split_type == 'scaffold' else 'rd'}.txt")

    vocab_path = vocab_map[tokenizer_type]
    if not vocab_path.startswith('seyonec/'):
        vocab_path = os.path.join(vocab_dir, vocab_path)

    if tokenizer_type == 'ais':
        tokenizer = AISTokenizer(vocab_path)
        bos_token, eos_token = '[CLS]', '[SEP]'
    elif tokenizer_type == 'char':
        tokenizer = CharLevelTokenizer(vocab_path)
        # Ensure special tokens are in the vocab
        for tok in ['[CLS]', '[SEP]']:
            if tok not in tokenizer.vocab:
                tokenizer.vocab[tok] = len(tokenizer.vocab)
        bos_token, eos_token = '[CLS]', '[SEP]'
    elif tokenizer_type.startswith('npbpe'):
        tokenizer = NPBPETokenizer(vocab_path)
        bos_token, eos_token = '[CLS]', '[SEP]'
    elif tokenizer_type == 'bpe':
        tokenizer = AutoTokenizer.from_pretrained(vocab_path)
        bos_token, eos_token = '<s>', '</s>'
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    bos_token_id = get_token_id(tokenizer, bos_token, tokenizer_type)
    eos_token_id = get_token_id(tokenizer, eos_token, tokenizer_type)
    
    train_loader = DataLoader(SMILESDataset(train_file, tokenizer), batch_size=32, shuffle=True, collate_fn=lambda b: collate_batch(b, tokenizer))
    val_loader = DataLoader(SMILESDataset(val_file, tokenizer), batch_size=32, shuffle=False, collate_fn=lambda b: collate_batch(b, tokenizer))
    test_loader = DataLoader(SMILESDataset(test_file, tokenizer), batch_size=32, shuffle=False, collate_fn=lambda b: collate_batch(b, tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "gpt":
        config_kwargs = dict(
            vocab_size=len(tokenizer),
            n_positions=512,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_embd * 4,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
        model = GPT2LMHeadModel(GPT2Config(**config_kwargs)).to(device)
    elif model_type in ["mamba1", "mamba2"]:
        config = MambaConfig(
            d_model=n_embd,
            n_layer=n_layer,
            d_intermediate=n_embd * 4,
            vocab_size=len(tokenizer),
            ssm_cfg={'layer': model_type.capitalize()},
            attn_layer_idx=[],
            attn_cfg={},
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True
        )
        model = MambaLMHeadModel(config).to(device)
    else:
        raise ValueError("Unsupported model type")

    optimizer = SAM(model.parameters(), optim.Adam, lr=lr, rho=0.05, weight_decay=0.0001)
    criterion = CrossEntropyLoss(ignore_index=0)

    project_name = f"{model_type}-{tokenizer_type}-{split_type}"
    wandb.init(project=project_name, config=config)

    best_loss, patience_counter = float('inf'), 0
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{max_epochs}", unit="batch") as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if inputs is None or targets is None:
                    continue  # Skip invalid batch

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

                pbar.update(1)
                if batch_idx % 10 == 0:
                    avg_batch_loss = epoch_loss / (batch_idx + 1)
                    wandb.log({"train_loss_batch": avg_batch_loss, "epoch": epoch + 1, "batch": batch_idx + 1})

        avg_train_loss = epoch_loss / num_batches
        train_perplexity = torch.exp(torch.tensor(avg_train_loss))

        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{max_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_ppl:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_perplexity": train_perplexity.item(),
            "val_loss": val_loss,
            "val_perplexity": val_ppl.item()
        })

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{project_name}_best_model.pth")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/5")
            if patience_counter >= 5:
                print("Early stopping triggered.")
                model.load_state_dict(torch.load(f"{project_name}_best_model.pth"))
                break
            
    test_loss, test_ppl = evaluate(model, test_loader, criterion, device)
    wandb.log({"test_loss": test_loss, "test_perplexity": test_ppl})
    print(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_ppl:.4f}")

    save_directory = f"./{project_name}"
    try:
        os.makedirs(save_directory, exist_ok=True)
        save_pretrained(model, tokenizer, save_directory)
        logging.info(f"Standard model files saved in {save_directory}.")
        safetensors_path = save_model_as_safetensors(model, save_directory)
        logging.info(f"Model saved successfully in custom safetensors format at {safetensors_path}.")
        artifact = wandb.Artifact(project_name, type='model')
        artifact.add_file(os.path.join(save_directory, 'pytorch_model.bin'))
        artifact.add_file(safetensors_path)
        wandb.log_artifact(artifact)
        logging.info("Model artifacts logged to WandB.")
    except Exception as e:
        logging.error(f"An error occurred while saving the model: {str(e)}")

    wandb.finish()
