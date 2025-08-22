import torch
import os
import csv
import time
from tqdm import tqdm
from torch.nn.functional import softmax, log_softmax
from transformers import AutoTokenizer, GPT2LMHeadModel
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from tokenisers import AISTokenizer, CharLevelTokenizer, NPBPETokenizer


def load_tokenizer(tokenizer_type, vocab_path=None):
    """
    Load a tokenizer based on the specified type and vocabulary path.

    Args:
        tokenizer_type (str): Type of tokenizer ('ais', 'char', 'npbpe', or 'bpe').
        vocab_path (str, optional): Path to the vocabulary file or HF model ID.

    Returns:
        tuple: (tokenizer object, BOS token string, EOS token string)
    
    Raises:
        ValueError: If an unknown tokenizer type is provided.
    """
    if tokenizer_type == 'ais':
        return AISTokenizer(vocab_path), '[CLS]', '[SEP]'
    elif tokenizer_type == 'char':
        return CharLevelTokenizer(vocab_path), '[CLS]', '[SEP]'
    elif tokenizer_type.startswith('npbpe'):
        return NPBPETokenizer(vocab_path), '[CLS]', '[SEP]'
    elif tokenizer_type == 'bpe':
        tokenizer = AutoTokenizer.from_pretrained(vocab_path)
        return tokenizer, '<s>', '</s>'
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

def load_model(model_name, model_type):
    """
    Load a pretrained model onto GPU in evaluation mode.

    Args:
        model_name (str): Hugging Face model ID.
        model_type (str): Type of model: 'gpt' or 'mamba'

    Returns:
        torch.nn.Module: Loaded model on CUDA.

    Raises:
        ValueError: If an unknown model type is provided.
    """
    if model_type == 'gpt':
        return GPT2LMHeadModel.from_pretrained(model_name).to("cuda").eval()
    elif model_type == 'mamba':
        return MambaLMHeadModel.from_pretrained(model_name).to("cuda").eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def infer_from_model_name(model_name):
    """
    Infer tokenizer type, model type, and vocabulary path from model name.

    Args:
        model_name (str): Name or path of the model.

    Returns:
        tuple: (tokenizer_type, model_type, vocab_path)
    
    Raises:
        ValueError: If the tokenizer type cannot be inferred.
    """
    name = os.path.basename(model_name).lower()

    # Dynamically resolve repo root
    repo_root = os.path.dirname(os.path.abspath(__file__))
    vocab_dir = os.path.join(repo_root, 'vocab_files')

    if 'ais' in name:
        tokenizer_type = 'ais'
        vocab_path = os.path.join(vocab_dir, 'ais_vocab.json')
    elif 'char' in name:
        tokenizer_type = 'char'
        vocab_path = os.path.join(vocab_dir, 'vocab.json')
    elif 'bpe' in name and 'npbpe' not in name:
        tokenizer_type = 'bpe'
        vocab_path = 'seyonec/PubChem10M_SMILES_BPE_450k'  # HF tokenizer ID
    elif 'npbpe60' in name:
        tokenizer_type = 'npbpe_60'
        vocab_path = os.path.join(vocab_dir, 'npbpe_60.json')
    elif 'npbpe1000' in name:
        tokenizer_type = 'npbpe_1000'
        vocab_path = os.path.join(vocab_dir, 'npbpe_1000.json')
    elif 'npbpe100' in name:
        tokenizer_type = 'npbpe_100'
        vocab_path = os.path.join(vocab_dir, 'npbpe_100.json')
    elif 'npbpe7924' in name:
        tokenizer_type = 'npbpe_7924'
        vocab_path = os.path.join(vocab_dir, 'npbpe_7924vocab.json')
    elif 'npbpe30k' in name:
        tokenizer_type = 'npbpe_30k'
        vocab_path = os.path.join(vocab_dir, 'npbpe_tokenizer.json')
    else:
        raise ValueError(f"Cannot infer tokenizer type and vocab path from model name: {model_name}")

    model_type = 'mamba' if name.startswith('m') else 'gpt'
    return tokenizer_type, model_type, vocab_path

def get_token_id(tokenizer, token, tokenizer_type):
    """
    Get the token ID for a given token using the specified tokenizer.

    Args:
        tokenizer: Tokenizer object.
        token (str): Token string.
        tokenizer_type (str): Type of tokenizer.

    Returns:
        int: Token ID.
    
    Raises:
        ValueError: If tokenizer type is unsupported.
    """
    if tokenizer_type.startswith("npbpe"):
        return tokenizer.tokenizer.token_to_id(token)
    elif tokenizer_type in ["ais", "char"]:
        return tokenizer.vocab[token]
    elif tokenizer_type == "bpe":
        return tokenizer.convert_tokens_to_ids(token)
    else:
        raise ValueError(f"Unsupported tokenizer type for token ID lookup: {tokenizer_type}")

def generate_molecules(model, tokenizer, num_molecules, max_length, temperature, filename,
                       bos_token, eos_token, batch_size=32, tokenizer_type=None):
    """
    Generate molecules using a model and save them with log-likelihoods to a CSV file.

    Args:
        model (torch.nn.Module): Language model for generation.
        tokenizer: Tokenizer used for decoding tokens to strings.
        num_molecules (int): Number of molecules to generate.
        max_length (int): Maximum sequence length for generation.
        temperature (float): Sampling temperature.
        filename (str): Path to output CSV file.
        bos_token (str): Beginning-of-sequence token.
        eos_token (str): End-of-sequence token.
        batch_size (int): Number of sequences to generate per batch.
        tokenizer_type (str): Type of tokenizer used (affects token ID lookup).

    Prints:
        Results are written to a file and summary stats are printed.
    """
    device = "cuda"
    file_exists = os.path.isfile(filename)
    file_empty = os.stat(filename).st_size == 0 if file_exists else False
    
    eos_token_id = get_token_id(tokenizer, eos_token, tokenizer_type)
    bos_token_id = get_token_id(tokenizer, bos_token, tokenizer_type) 

    total_start_time = time.time()
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists or file_empty:
            writer.writerow(['Molecule', 'Log-Likelihood'])

        molecules, loglikelihoods = [], []
        total_time = 0

        with tqdm(total=num_molecules, desc="Generating molecules", unit="molecule") as pbar:
            with torch.no_grad():
                for _ in range(0, num_molecules, batch_size):
                    current_batch_size = min(batch_size, num_molecules - len(molecules))
                    input_ids = torch.tensor([bos_token_id] * current_batch_size, device=device).unsqueeze(1)
                    generated = input_ids
                    batch_log_likelihoods = [0] * current_batch_size
                    active_mask = torch.ones(current_batch_size, dtype=torch.bool, device=device)
                    batch_start_time = time.time()

                    for _ in range(max_length):
                        outputs = model(input_ids=generated)
                        logits = outputs.logits[:, -1, :] / temperature
                        log_probs = log_softmax(logits, dim=-1)
                        probabilities = softmax(logits, dim=-1)
                        next_tokens = torch.multinomial(probabilities, num_samples=1)

                        next_tokens = next_tokens * active_mask.unsqueeze(1) + eos_token_id * (~active_mask).unsqueeze(1)
                        generated = torch.cat((generated, next_tokens), dim=1)

                        for i in range(current_batch_size):
                            if active_mask[i]:
                                log_prob = log_probs[i, next_tokens[i].item()]
                                batch_log_likelihoods[i] += log_prob.item()

                        active_mask &= (next_tokens.squeeze(1) != eos_token_id)
                        if not active_mask.any():
                            break

                    total_time += time.time() - batch_start_time

                    for i in range(current_batch_size):
                        mol = tokenizer.decode(generated[i].tolist(), skip_special_tokens=True)
                        molecules.append(mol)
                        loglikelihoods.append(batch_log_likelihoods[i])

                        if len(molecules) >= batch_size:
                            writer.writerows(zip(molecules, loglikelihoods))
                            file.flush()
                            molecules.clear()
                            loglikelihoods.clear()
                    pbar.update(current_batch_size)

                if molecules:
                    writer.writerows(zip(molecules, loglikelihoods))
                    file.flush()

    avg_time = total_time / num_molecules
    print(f"Average generation time per molecule: {avg_time:.4f}s")
    print(f"Total time: {time.time() - total_start_time:.4f}s")
    print(f"Saved to {filename}")

    
def run_generation(config):
    """
    Run the molecule generation pipeline from a configuration dictionary.

    Args:
        config (dict): Dictionary containing keys:
            - 'model_name' (str): Path or HF ID of model.
            - 'num_mols' (int): Number of molecules to generate.
            - 'max_length' (int): Maximum length of each molecule.
            - 'temperature' (float): Sampling temperature.
            - 'outfile' (str): Output CSV file path.
    """

    tokenizer_type, model_type, vocab_path = infer_from_model_name(config["model_name"])
    tokenizer, bos, eos = load_tokenizer(tokenizer_type, vocab_path)
    model = load_model(config["model_name"], model_type)

    generate_molecules(
        model=model,
        tokenizer=tokenizer,
        num_molecules=config["num_mols"],
        max_length=config["max_length"],
        temperature=config["temperature"],
        filename=config["outfile"],
        bos_token=bos,
        eos_token=eos,
        batch_size=32,
        tokenizer_type=tokenizer_type
    )

