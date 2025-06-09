# 🧪 CLMs for Natural Products

- ✅ Natural Products Molecule Generation and Property Prediction 
- 🔍 Model Pre-training Hyperparameter Search  
- 🧠 Model Pre-training for Mamba, Mamba2, and GPT
- 🧬 Fine-tuning for Property Prediction  

All tasks are managed through `main.py` and can be executed using a simple bash script.

---

## 🗂️ Directory Structure

```
main.py                  # Entry point
mol_generation.py        # NP molecule generation 
hpsearch.py              # Model pre-training hyperparameter search  
pretraining.py           # Model pre-training for Mamba, Mamba2, and GPT
finetuning.py            # Fine-tuning on property prediction tasks
sam.py                   # SAM implementation from UU-Mamba (arXiv:2402.03394)
tokenisers.py            # Custom tokenizers implementation 
data/                    # Contains pre-training 1M NPs and downstream task data files
molformer_n_chemberta_2/ # Contains MolFormer and ChemBERTa-2 fine-tuning code 
vocab_files/             # Contains vocab.json files for all custom tokenizers 
lsv_cluster_files/       # Contains cluster-related setup
├── mamba.dockerfile     # Dockerfile for Mamba training environment
├── run_experiments.sh   # Shell script to run experiments using main.py 
└── run_experiments.sub  # Cluster job submission script


```

---

## 📦 Environment Setup

You can run this project either directly with Python or using Docker (**recommended**).

### 🔧 Option 1: Docker (Recommended)

1. **Build the image:**


2. **Run the container:**


## ⚙️ How to Run Tasks

All tasks are executed via `main.py`. A helper script `run_pretrain.sh` is provided to simplify usage.

### 🔧 Script Usage

```bash
bash run_pretrain.sh YOUR_WANDB_KEY
```

Uncomment the block corresponding to the task you want to run.

---

### 1. Molecule Generation

```bash
python3 main.py \
  --task generate \
  --num_mols 50 \
  --temperature 0.7 \
  --max_length 256 \
  --model_names rozariwang/M2-NPBPE1000-rds
```

---

### 2. Hyperparameter Search

```bash
python3 main.py \
  --task hpsearch \
  --hp_model GPT \
  --hp_tokenizer AIS \
  --hp_split random
```

---

### 3. Pretraining (requires WandB key)

```bash
python3 main.py \
  --task pretrain \
  --wandb_key YOUR_WANDB_KEY \
  --pt_model GPT \
  --pt_tokenizer NPBPE1000 \
  --pt_split random \
  --pt_n_embd 256 \
  --pt_n_layer 8 \
  --pt_lr 1e-4 \
  --pt_n_head 4
```

Use `--pt_n_head None` for non-GPT models.

---

### 4. Fine-tuning

Fine-tuning and evaluation for 3 downstream classification tasks using 48 NP 
pretrained models (3 model types (GPT, Mamba, or Mamba2) * 8 tokenizers * 2 data split methods)

Options:
- task: "finetune"
- sub_task: "anti_cancer", "peptides", "tastes"
- model_split: "sfs" or "rds"  (how pre-training 1M NPs data is split)
- data_split: "sf" or "rd"
```bash
python3 main.py \
  --task finetune \
  --sub_task peptides \
  --model_split sfs \
  --data_split sf
```

---

### 5. Fine-tuning ChemBERTa-2

Fine-tuning ChemBERTa-2 MLM on 1M NPs
```bash
python3 ChemBERTa2_MLM_Finetune_on_1M_NPs.py 
```

Fine-tuning ChemBERTa-2 on property prediction tasks \
Options: 
- task: "chemberta"
- chemberta_model_type: "mlm" (original model), "mtr" (original model), "mlm-finetuned" (fine-tuned on 1M NPs) 
- sub_task: "anti_cancer" or "peptides"
- data_split: "rd" or "sf"
```bash
python3 main.py \
  --task chemberta \
  --chemberta_model_type mtr \
  --sub_task anti_cancer \
  --data_split sf
```

### 6. Fine-tuning MolFormer

Fine-tuning MolFormer on 1M NPs
```bash
python3 main.py \
  --task molformer_1M_NPs \
  --wandb_key "$1"
```

Fine-tuning MolFormer on property prediction tasks \
Options: 
- task: "molformer" (original model) or "molformer-finetuned" (fine-tuned on 1M NPs)
- sub_task: "anti_cancer" or "peptides"
- data_split: "rd" or "sf"
```bash
python3 main.py \
  --task molformer \
  --sub_task peptides \
  --data_split rd
```
