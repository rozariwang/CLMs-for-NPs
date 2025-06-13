#!/bin/bash
# run_experiments.sh
#
# This script provides examples for running molecule generation, hyperparameter search,
# pretraining, and fine-tuning. Please specify your Weights & Biases API key (required for some tasks) 
# as an argument in the .sub file.  This script acts as a template. Uncomment the desired block 
# to run a specific task. All tasks are executed via main.py, which dispatches based on the `--task` argument.

##############################
# 1. Molecule Generation Task
##############################

#python3 /nethome/hhwang/hhwang/CLMs-for-NPs/main.py \
#   --task generate \
#   --num_mols 50 \
#   --temperature 0.7 \
#   --max_length 256 \
#   --model_names rozariwang/M2-NPBPE1000-rds

##############################
# 2. Hyperparameter Search Task
##############################

#python3 /nethome/hhwang/hhwang/CLMs-for-NPs/main.py \
#   --task hpsearch \
#   --hp_model GPT \
#   --hp_tokenizer AIS \
#   --hp_split random

##############################
# 3. Pretraining Task 
# 3 model types * 8 tokenizers * 2 data splits = 48 combinations
# Set hyperparameters form the hyperparameter search result. 
# Set "pt_n_head" to "None" for Mamba models. 
##############################

python3 /nethome/hhwang/hhwang/CLMs-for-NPs/main.py \
   --task pretrain \
   --wandb_key "$1" \
   --pt_model GPT \
   --pt_tokenizer NPBPE1000 \
   --pt_split random \
   --pt_n_embd 256 \
   --pt_n_layer 8 \
   --pt_lr 1e-4 \
   --pt_n_head 4

##############################
# 4. Fine-tuning Task 
# (Property prediction using the 48 NP-pretrained models.)
##############################

#python3 /nethome/hhwang/hhwang/CLMs-for-NPs/main.py \
#  --task finetune \
#  --sub_task peptides \
#  --model_split sfs \
#  --data_split sf

##############################
# 5. Fine-tuning ChemBERTa-2 
##############################

### Fine-tuning ChemBERTa-2 MLM on 1M NPs ###

#python3 /nethome/hhwang/hhwang/CLMs-for-NPs/ChemBERTa2_MLM_Finetune_on_1M_NPs.py

### Fine-tuning ChemBERTa-2 on property prediction tasks ###

#python3 /nethome/hhwang/hhwang/CLMs-for-NPs/main.py \
#  --task chemberta \
#  --chemberta_model_type mlm-finetuned \
#  --sub_task peptides \
#  --data_split sf

##############################
# 5. Fine-tuning MolFormer
##############################

### Fine-tuning MolFormer on 1M NPs ###

#python3 /nethome/hhwang/hhwang/CLMs-for-NPs/main.py \
#  --task molformer_1M_NPs \
#  --wandb_key "$1"

### Fine-tuning MolFormer on property prediction tasks ###

#python3 /nethome/hhwang/hhwang/CLMs-for-NPs/main.py \
#  --task molformer \
#  --molformer_variant molformer-finetuned \
#  --sub_task peptides \
#  --data_split rd



