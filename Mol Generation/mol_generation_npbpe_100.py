import torch
from transformers import PreTrainedTokenizer, GPT2LMHeadModel
from torch.nn.functional import softmax, log_softmax
from torch.nn import CrossEntropyLoss
import csv
import json
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import os
import time
from tqdm import tqdm
from tokenisers import NPBPETokenizer
from huggingface_hub import login

login(token="your_huggingface_token")  # Replace with your actual Hugging Face API token


# Initialize tokenizer and model
device = "cuda"
model_name = "rozariwang/M2-npbpe100-sfs"
tokenizer = NPBPETokenizer("./hhwang/shiawaseda/Datasets/npbpe_100.json")
model = MambaLMHeadModel.from_pretrained(model_name).to(device)
#model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()


def generate_molecules(model, tokenizer, num_molecules, max_length, temperature, filename, batch_size=32):
    model.eval()
    eos_token_id = tokenizer.tokenizer.token_to_id('[SEP]')  # End of Sequence Token
    bos_token_id = tokenizer.tokenizer.token_to_id('[CLS]')  # Start of Sequence Token

    # Check if file exists and if it is empty
    file_exists = os.path.isfile(filename)
    file_empty = os.stat(filename).st_size == 0 if file_exists else False

    # Start total timer for the entire generation process
    total_start_time = time.time()
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header only if the file did not exist or it was empty
        if not file_exists or file_empty:
            writer.writerow(['Molecule', 'Log-Likelihood'])
            
        molecules = []
        loglikelihoods = []
        total_time = 0  # Total time to track generation time
        
        # Wrap the outer loop in tqdm for a progress bar
        with tqdm(total=num_molecules, desc="Generating molecules", unit="molecule") as pbar:
            with torch.no_grad():
                for _ in range(0, num_molecules, batch_size):
                    current_batch_size = min(batch_size, num_molecules - len(molecules))  # Handle last smaller batch
                    input_ids = torch.tensor([bos_token_id] * current_batch_size, device=device).unsqueeze(1)
                    generated = input_ids
                    batch_log_likelihoods = [0] * current_batch_size  # Initialize log-likelihoods for each molecule in the batch

                    # Start timer for the batch
                    batch_start_time = time.time()
                    eos_token_id = tokenizer.tokenizer.token_to_id('[SEP]')  # Use only the EOS token as the termination criterion

                    # Initialize a binary mask to track active sequences (sequences that haven't reached [SEP])
                    active_mask = torch.ones(current_batch_size, dtype=torch.bool, device=device)

                    for _ in range(max_length):
                        outputs = model(input_ids=generated)
                        logits = outputs.logits[:, -1, :] / temperature
                        log_probs = log_softmax(logits, dim=-1)
                        probabilities = softmax(logits, dim=-1)
                        next_tokens = torch.multinomial(probabilities, num_samples=1)

                        # Mask out inactive sequences by forcing them to generate [SEP] repeatedly
                        next_tokens = next_tokens * active_mask.unsqueeze(1) + eos_token_id * (~active_mask).unsqueeze(1)
    
                        generated = torch.cat((generated, next_tokens), dim=1)

                        # Update log-likelihoods for each active sequence in the batch
                        for i in range(current_batch_size):
                            if active_mask[i]:
                                log_prob = log_probs[i, next_tokens[i].item()]
                                batch_log_likelihoods[i] += log_prob.item()

                        # Update the active mask for sequences that have reached [SEP]
                        active_mask &= (next_tokens.squeeze(1) != eos_token_id)

                        # Stop the loop early if all sequences are inactive
                        if not active_mask.any():
                            break

                    # End timer for the batch
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time  # Time taken for the batch
                    total_time += batch_time  # Accumulate the total time for averaging later

                    # Decode and save each molecule in the batch
                    for i in range(current_batch_size):
                        molecule = tokenizer.decode(generated[i].tolist(), skip_special_tokens=True)
                        molecules.append(molecule)
                        loglikelihoods.append(batch_log_likelihoods[i])

                        # Save once the molecules reach a batch size to reduce file I/O
                        if len(molecules) >= batch_size:
                            writer.writerows(zip(molecules, loglikelihoods))
                            file.flush()
                            molecules.clear()
                            loglikelihoods.clear()

                # Write any remaining molecules after final loop
                if molecules:
                    writer.writerows(zip(molecules, loglikelihoods))
                    file.flush()

        
        # End total timer for the entire generation process
        total_end_time = time.time()
        total_generation_time = total_end_time - total_start_time  # Total time for all molecules
        
        # Calculate and print average generation time per molecule
        average_time_per_molecule = total_time / num_molecules
        print(f"Average generation time per molecule: {average_time_per_molecule:.4f} seconds")
        print(f"Total time taken to generate all molecules: {total_generation_time:.4f} seconds")
        print(f"Data saved to {filename}.")


# Generate molecules and store in a CSV file
num_molecules = 100000
temperature = 1
file_name = 'M2_npbpe100_sfs_generated_mol_100000.csv'
generate_molecules(model, tokenizer, num_molecules=num_molecules, max_length=512, temperature=temperature, filename=file_name, batch_size=32)