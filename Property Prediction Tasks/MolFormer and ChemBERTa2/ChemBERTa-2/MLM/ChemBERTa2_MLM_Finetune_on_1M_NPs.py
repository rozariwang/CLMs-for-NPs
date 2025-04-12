import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import optuna
import random

class SMILESDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r') as file:
            self.data = [line.strip() for line in file if line.strip()]  # Store raw text only

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        tokens = self.tokenizer(
            line,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }

def evaluate(model, data_loader, device):
    model.eval()
    eval_loss = 0.0
    eval_steps = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs.loss.item()
            eval_steps += 1

    avg_loss = eval_loss / eval_steps
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity

def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    
    # Print current hyperparameters being tested
    print(f"\nStarting hyperparameter trial with: learning_rate={learning_rate}, batch_size={batch_size}")

    # Initialize model, optimizer, and data loaders
    model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(search_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(search_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    # Training loop for exactly 5 epochs
    for epoch in range(5):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_perplexity = evaluate(model, val_loader, device)

        # Print progress for this epoch
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}, Validation Perplexity = {val_perplexity:.4f}")

    # After 5 epochs, return the validation loss from the last epoch
    print(f"Trial complete. Final Validation Loss = {val_loss:.4f}")
    return val_loss


# Load datasets
max_length = 512
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

full_train_dataset = SMILESDataset("./hhwang/shiawaseda/Datasets/train_rd.txt", tokenizer, max_length=max_length)
full_val_dataset = SMILESDataset("./hhwang/shiawaseda/Datasets/val_rd.txt", tokenizer, max_length=max_length)
full_test_dataset = SMILESDataset("./hhwang/shiawaseda/Datasets/test_rd.txt", tokenizer, max_length=max_length)

# Create 5% subsets for hyperparameter search
train_size = int(0.05 * len(full_train_dataset))
val_size = int(0.05 * len(full_val_dataset))

train_indices = random.sample(range(len(full_train_dataset)), train_size)
val_indices = random.sample(range(len(full_val_dataset)), val_size)

search_train_dataset = Subset(full_train_dataset, train_indices)
search_val_dataset = Subset(full_val_dataset, val_indices)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # Enable Masked Language Modeling
    mlm_probability=0.15,  # Probability of masking tokens
)


# Hyperparameter tuning
print("\nStarting hyperparameter search...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Retrieve the best hyperparameters
best_trial = study.best_trial
best_learning_rate = best_trial.params["learning_rate"]
best_batch_size = best_trial.params["batch_size"]

print("\nHyperparameter search complete!")
print(f"Best learning rate: {best_learning_rate}")
print(f"Best batch size: {best_batch_size}")

# Train with the best hyperparameters
print("\nStarting fine-tuning with the best hyperparameters...")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=best_learning_rate)

train_loader = DataLoader(full_train_dataset, batch_size=best_batch_size, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(full_val_dataset, batch_size=best_batch_size, shuffle=False, collate_fn=data_collator)

best_val_loss = float("inf")
patience = 5
epochs_no_improve = 0

for epoch in range(100):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    val_loss, val_perplexity = evaluate(model, val_loader, device)
    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}, Validation Perplexity = {val_perplexity:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./ChemBERTa2_finetuned_model.pth")
    else:
        epochs_no_improve += 1
        print(f"Patience count: {epochs_no_improve}/{patience}")

    if epochs_no_improve >= patience:
        print("Stopping early due to no improvement.")
        break

# Load best model for testing
print("\nTesting the best model...")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(device)
model.load_state_dict(torch.load("./ChemBERTa2_finetuned_model.pth"))

# Test evaluation
test_loader = DataLoader(full_test_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)
test_loss, test_perplexity = evaluate(model, test_loader, device)
print("Test Loss:", test_loss)
print("Test Perplexity:", test_perplexity.item())