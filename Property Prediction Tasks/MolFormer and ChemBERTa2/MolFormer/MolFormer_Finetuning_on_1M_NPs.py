import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import random
import wandb
from tqdm import tqdm

wandb.login(key="your_wandb_api_key") 


# Initialize wandb 
wandb.init(project="MoLFormer_finetuning", name="MoLFormer_finetuning_run", config={
    "learning_rate": 1.1532933619007571e-05,
    "batch_size": 8,
    "epochs": 100,
    "patience": 5,
    "max_length": 512,
    "mlm_probability": 0.15
})

class SMILESDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r') as file:
            self.data = [line.strip() for line in file if line.strip()]

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
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": tokens["input_ids"].squeeze(0),
        }

def evaluate(model, data_loader, device):
    model.eval()
    eval_loss = 0.0
    eval_steps = 0

    # Wrap the evaluation loop with tqdm
    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs.loss.item()
            eval_steps += 1

    avg_loss = eval_loss / eval_steps
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity

max_length = 512
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

full_train_dataset = SMILESDataset("./hhwang/shiawaseda/Datasets/train_rd.txt", tokenizer, max_length=max_length)
full_val_dataset = SMILESDataset("./hhwang/shiawaseda/Datasets/val_rd.txt", tokenizer, max_length=max_length)
full_test_dataset = SMILESDataset("./hhwang/shiawaseda/Datasets/test_rd.txt", tokenizer, max_length=max_length)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# Use the best hyperparameters found
best_learning_rate = 1.1532933619007571e-05
best_batch_size = 8

print("\nStarting fine-tuning with the best hyperparameters...")
model = AutoModelForMaskedLM.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=best_learning_rate)

# Optionally, have wandb watch the model for logging gradients and parameters
wandb.watch(model, log="all")

train_loader = DataLoader(full_train_dataset, batch_size=best_batch_size, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(full_val_dataset, batch_size=best_batch_size, shuffle=False, collate_fn=data_collator)

best_val_loss = float("inf")
patience = 5
epochs_no_improve = 0
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    # Use tqdm to track progress for each training batch in the epoch
    train_batches = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
    for batch in train_batches:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # Optionally update tqdm's postfix with the current loss
        train_batches.set_postfix(loss=f"{loss.item():.4f}")

    train_loss /= len(train_loader)
    val_loss, val_perplexity = evaluate(model, val_loader, device)
    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}, Validation Perplexity = {val_perplexity:.4f}")
    
    # Log epoch metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_perplexity": val_perplexity.item()
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "./MoLFormer_finetuned_model.pth")
    else:
        epochs_no_improve += 1
        print(f"Patience count: {epochs_no_improve}/{patience}")

    if epochs_no_improve >= patience:
        print("Stopping early due to no improvement.")
        break

print("\nTesting the best model...")
model.load_state_dict(torch.load("./MoLFormer_finetuned_model.pth"))
test_loader = DataLoader(full_test_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)
test_loss, test_perplexity = evaluate(model, test_loader, device)
print("Test Loss:", test_loss)
print("Test Perplexity:", test_perplexity.item())

wandb.log({
    "test_loss": test_loss,
    "test_perplexity": test_perplexity.item()
})

wandb.finish()