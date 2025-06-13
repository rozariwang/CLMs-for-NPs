import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from sam import SAM
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from tokenisers import AISTokenizer, CharLevelTokenizer, NPBPETokenizer

def run_hpsearch(config):
    """
    Run random hyperparameter search for all model-tokenizer pair.

    This function trains either a GPT2-based or Mamba-based model on tokenized SMILES strings
    using different configurations of model hyperparameters. For each unique combination, the model
    is trained for 5 epochs, and its performance is evaluated on a held-out validation set.

    The hyperparameter combinations are sampled randomly within half of the entire search space (36 for GPT, 12 for Mamba). 
    The model achieving the lowest validation loss is reported as the best configuration.

    Args:
        config (dict): Dictionary with keys:
            - 'tokenizer' (str): Tokenizer type ('ais', 'char', 'bpe', 'npbpe*').
            - 'model' (str): Model type ('gpt', 'mamba1', 'mamba2').
            - 'split' (str): Data split type ('scaffold' or 'random').

    Prints:
        - Epoch-level training and validation loss per configuration.
        - A summary DataFrame of all configurations and their validation losses.
        - The best hyperparameter configuration based on lowest validation loss.
    """
    def load_tokenizer(tokenizer_type, vocab_path):
        if tokenizer_type == 'ais':
            return AISTokenizer(vocab_path)
        elif tokenizer_type == 'char':
            return CharLevelTokenizer(vocab_path)
        elif tokenizer_type.startswith('npbpe'):
            return NPBPETokenizer(vocab_path)
        elif tokenizer_type == 'bpe':
            return AutoTokenizer.from_pretrained(vocab_path)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    class SMILESDataset(Dataset):
        """
        Custom PyTorch dataset for loading SMILES strings and tokenizing them.
        """
        def __init__(self, file_path, tokenizer, max_length=512):
            """
            Args:
                file_path (str): Path to text file containing SMILES strings.
                tokenizer (Tokenizer): Tokenizer to encode SMILES strings.
                max_length (int): Maximum sequence length.
            """
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
            """Returns the total number of SMILES strings."""
            return len(self.smiles)

        def __getitem__(self, idx):
            """
            Retrieves and tokenizes a SMILES string at a given index.

            Args:
                idx (int): Index of the item to retrieve.

            Returns:
                torch.Tensor: Tokenized SMILES string as a tensor.
            """
            smiles_string = self.smiles[idx]
            tokenized = self.tokenizer.encode(smiles_string, add_special_tokens=True, max_length=self.max_length, truncation=True)
            tensor = torch.tensor(tokenized, dtype=torch.long)
            return tensor

    def collate_batch(batch):
        """
        Collate function for DataLoader to pad sequences and create input-target pairs.

        Args:
            batch (List[Tensor]): List of tokenized SMILES sequences.

        Returns:
            Tuple[Tensor, Tensor]: Padded input and target sequences.
        """
        padding_value = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        batch_padded = pad_sequence(batch, batch_first=True, padding_value=padding_value)
        inputs = batch_padded[:, :-1]
        targets = batch_padded[:, 1:]
        return inputs.long(), targets.long()

    def get_dataloaders(tokenizer, split_type):
        """
        Load and return train and validation DataLoaders for SMILES data.

        Args:
            tokenizer: Tokenizer to apply to SMILES strings.
            split_type (str): Data split type ('scaffold' or 'random').

        Returns:
            Tuple[DataLoader, DataLoader]: Train and validation DataLoaders.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data", "1M_NPs")

        if split_type == "scaffold":
            train_file = os.path.join(data_dir, "train_sf.txt")
            val_file = os.path.join(data_dir, "val_sf.txt")
        else:
            train_file = os.path.join(data_dir, "train_rd.txt")
            val_file = os.path.join(data_dir, "val_rd.txt")

        total_train_dataset = SMILESDataset(train_file, tokenizer, max_length=512)
        total_val_dataset = SMILESDataset(val_file, tokenizer, max_length=512)

        train_subset_indices = torch.randperm(len(total_train_dataset))[:max(1, int(0.05 * len(total_train_dataset)))]
        train_subset_dataset = Subset(total_train_dataset, train_subset_indices)
        val_subset_indices = torch.randperm(len(total_val_dataset))[:max(1, int(0.05 * len(total_val_dataset)))]
        val_subset_dataset = Subset(total_val_dataset, val_subset_indices)

        train_loader = DataLoader(train_subset_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_subset_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

        return train_loader, val_loader

    def evaluate(model, data_loader, criterion, device):
        """
        Evaluate model performance on a validation set.

        Args:
            model (nn.Module): Trained model.
            data_loader (DataLoader): DataLoader for validation data.
            criterion (Loss): Loss function.
            device (torch.device): Computation device.

        Returns:
            float: Average loss on the validation set.
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
            
        return eval_loss / num_batches

    def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, hyperparams):
        """
        Train the model for a specified number of epochs.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (Optimizer): SAM optimizer.
            criterion (Loss): Loss function.
            device (torch.device): Computation device.
            num_epochs (int): Number of training epochs.
            hyperparams (dict): Dictionary of current hyperparameters.
        """
        print(f"Training with hyperparameters: {hyperparams}")
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(device).long()
                targets = targets.to(device).view(-1).long()

                def closure():
                    optimizer.zero_grad() # Reset gradients
                    outputs = model(inputs) # Forward pass
                    logits = outputs.logits  # Access logits
                    logits = logits.view(-1, logits.size(-1)) # Reshape logits for loss calculation
                    targets_res = targets.view(-1)  # Reshape targets
                    loss = criterion(logits, targets_res)  # Calculate loss
                    loss.backward()  # Backward pass (calculate gradients)
                    return loss
                
                # Perform the SAM optimizer step
                loss = closure() # compute the loss and gradients
                optimizer.step(closure)  # perform the optimizer step with SAM
                # Update epoch loss
                epoch_loss += loss.item()
                num_batches += 1

            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss / num_batches:.4f}, Val Loss = {val_loss:.4f}")

    tokenizer_type = config['tokenizer'].lower()
    model_type = config['model'].lower()
    split_type = config['split'].lower()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_dir = os.path.join(script_dir, 'vocab_files')

    vocab_map = {
        'ais': 'ais_vocab.json',
        'char': 'vocab.json',
        'bpe': 'seyonec/PubChem10M_SMILES_BPE_450k',
        'npbpe60': 'npbpe_60.json',
        'npbpe100': 'npbpe_100.json',
        'npbpe1000': 'npbpe_1000.json',
        'npbpe7924': 'npbpe_7924vocab.json',
        'npbpe30k': 'npbpe_tokenizer.json'
    }

    vocab_path = vocab_map[tokenizer_type]
    if not vocab_path.startswith('seyonec/'):
        vocab_path = os.path.join(vocab_dir, vocab_path)

    tokenizer = load_tokenizer(tokenizer_type, vocab_path)
    train_loader, val_loader = get_dataloaders(tokenizer, split_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    tested_combinations = set()

    if model_type == 'gpt':
        n_embd_options = [256, 512]
        n_layer_options = [4, 8, 12]
        n_head_options = [2, 4, 8]
        lr_options = [0.001, 0.005, 0.0001, 0.0005]
        num_search_iters = 36
    else:
        n_embd_options = [256, 512]
        n_layer_options = [4, 8, 12]
        n_head_options = [None]  # Mamba doesn't use heads
        lr_options = [0.001, 0.005, 0.0001, 0.0005]
        num_search_iters = 12

    num_epochs = 5

    for _ in range(num_search_iters):
        while True:
            n_embd = random.choice(n_embd_options)
            n_layer = random.choice(n_layer_options)
            n_head = random.choice(n_head_options)
            lr = random.choice(lr_options)
            combo = (n_embd, n_layer, n_head, lr)
            if combo not in tested_combinations:
                tested_combinations.add(combo)
                break

        if model_type == 'gpt':
            model_config = GPT2Config(
                vocab_size=len(tokenizer) if hasattr(tokenizer, '__len__') else tokenizer.vocab_size,
                n_positions=512,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_inner=n_embd * 4,
                bos_token_id=2,
                eos_token_id=3
            )
            model = GPT2LMHeadModel(model_config).to(device)
        else:
            ssm_layer = 'Mamba2' if model_type == 'mamba2' else 'Mamba1'
            model_config = MambaConfig(
                d_model=n_embd,
                n_layer=n_layer,
                d_intermediate=n_embd * 4,
                vocab_size=len(tokenizer) if hasattr(tokenizer, '__len__') else tokenizer.vocab_size,
                ssm_cfg={'layer': ssm_layer},
                attn_layer_idx=[],
                attn_cfg={},
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True
            )
            model = MambaLMHeadModel(model_config).to(device)

        base_optimizer = optim.Adam
        optimizer = SAM(model.parameters(), base_optimizer, lr=lr, rho=0.05, weight_decay=0.0001)
        criterion = CrossEntropyLoss(ignore_index=0)

        hyperparams = {
            'n_embd': n_embd,
            'n_layer': n_layer,
            'n_head': n_head,
            'lr': lr
        }

        train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, hyperparams)
        val_loss = evaluate(model, val_loader, criterion, device)

        results.append({**hyperparams, 'val_loss': val_loss})

    results_df = pd.DataFrame(results)
    print("\nRandom search results summary:")
    print(results_df)
    best_result = results_df.loc[results_df['val_loss'].idxmin()]
    print(f"\nBest hyperparameter set for Model: {config['model']} | Tokenizer: {config['tokenizer']} | Data Split: {config['split']}:")
    print(best_result.to_dict())