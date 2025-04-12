from transformers import PreTrainedTokenizer
from tokenizers import Tokenizer
import atomInSmiles
import json
from rdkit import Chem


class CharLevelTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.ids_to_tokens = {id: token for token, id in self.vocab.items()}

    def tokenize(self, text):
        return list(text)  # Tokenize the text into a list of characters

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]

    def encode(self, text, add_special_tokens=False, max_length=None, truncation=False):
        tokens = self.tokenize(text)
        token_ids = self.convert_tokens_to_ids(tokens)
        if add_special_tokens:
            token_ids = [self.vocab['[CLS]']] + token_ids + [self.vocab['[SEP]']]
        if truncation and max_length:
            token_ids = token_ids[:max_length]
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        special_tokens = [
            self.vocab.get('[CLS]'),
            self.vocab.get('[SEP]'),
            self.vocab.get('[PAD]'),
            self.vocab.get('[UNK]')
        ]
    
        if skip_special_tokens:
            token_ids = [id for id in token_ids if id not in special_tokens]
            
        return ''.join(self.ids_to_tokens.get(id, '[UNK]') for id in token_ids)

    def __len__(self):
        return len(self.vocab)


class AISTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.ids_to_tokens = {id: token for token, id in self.vocab.items()}
    
    def tokenize(self, smiles):
        try:
            # Validate SMILES using RDKit before passing to atomInSmiles
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Invalid SMILES skipped: {smiles}")
                return []  # Return empty token list

            # Tokenize using atomInSmiles
            ais_token_string = atomInSmiles.encode(smiles)  # Returns a space-separated string of tokens
            return ais_token_string.split()

        except Exception as e:
            print(f"Error tokenizing SMILES '{smiles}': {e}")
            return []  # Return empty token list instead of crashing

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]  # Map tokens to their IDs

    def encode(self, smiles, add_special_tokens=True, max_length=None, truncation=False):
        tokens = self.tokenize(smiles)
        token_ids = self.convert_tokens_to_ids(tokens)
        if add_special_tokens:
            token_ids = [self.vocab.get('[CLS]')] + token_ids + [self.vocab.get('[SEP]')]
        if truncation and max_length:
            token_ids = token_ids[:max_length]
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [self.ids_to_tokens.get(id, '[UNK]') for id in token_ids]  # Convert token IDs to tokens, replacing unknown IDs with '[UNK]'
        if skip_special_tokens:
            special_tokens = {'[CLS]', '[SEP]', '[PAD]', '[UNK]'}
            tokens = [token for token in tokens if token not in special_tokens]  # Filter out special tokens

        smiles_token_string = ' '.join(tokens)  # Join the tokens into a space-separated string for decoding
    
        try:
            # Attempt to decode using atomInSmiles
            decoded_smiles = atomInSmiles.decode(smiles_token_string)  # Decode the SMILES string into its chemical structure representation
        except IndexError as e:
            print(f"Decoding error with SMILES string '{smiles_token_string}': {e}. Skipping.")
            return None  # Return None or an empty string to indicate failure

        return decoded_smiles.replace(' ', '') if decoded_smiles else None

    def __len__(self):
        return len(self.vocab) 

class NPBPETokenizer(PreTrainedTokenizer):
    def __init__(self, tokenizer_file):
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
    
    def tokenize(self, text):
        # Tokenize using the pre-trained tokenizer
        encoding = self.tokenizer.encode(text)
        return encoding.tokens  # Return list of tokens

    def convert_tokens_to_ids(self, tokens):
        # Convert tokens to IDs using the pre-trained tokenizer's encoding
        encoding = self.tokenizer.encode(" ".join(tokens))
        return encoding.ids

    def encode(self, text, add_special_tokens=True, max_length=None, truncation=False):
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        if add_special_tokens:
            token_ids = [self.tokenizer.token_to_id("[CLS]")] + token_ids + [self.tokenizer.token_to_id("[SEP]")]
        if truncation and max_length:
            token_ids = token_ids[:max_length]
        return token_ids

    def decode(self, token_ids, skip_special_tokens=False):
        if skip_special_tokens: # Define special token IDs
            special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
            special_token_ids = {self.tokenizer.token_to_id(token) for token in special_tokens if self.tokenizer.token_to_id(token) is not None}
            token_ids = [id for id in token_ids if id not in special_token_ids] 
        return self.tokenizer.decode(token_ids)

    def __len__(self):
        return self.tokenizer.get_vocab_size() 

