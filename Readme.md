/Rosalie-Masters-thesis-repository
│
├── /Data/                             # Pre-training and Downstream Task Datasets
│   ├── 1M NP Pre-training Data/       # Random and Scaffold Split 1M NPs pre-training data
│   └── Downstream_Data/               # Random and Scaffold Split 5x5 CV Downstream Task Datasets
│
├── /lsv_cluster_files/                # LSV cluster Docker and .sh/.sub files
│
├── /Mol Generation/                   # Molecule Generation Code 
│
├── /NP CLMs Pretraining/              # Code for pretraining the 48 NP CLMs 
│   ├── Model HPsearch/                # Hyperparameter search code
│   └── Model Pretraining/             # Pretraining code
│
├── /Property Prediction Tasks/        # Code for finetuning on property rediction tasks
│   ├── Anti-Cancer/                   # Finetuning for Anti-Cancer tasks (48 NP models)
│   ├── FourTastes/                    # Finetuning for FourTastes tasks (48 NP models)
│   ├── MolFormer and ChemBERTa2/      # Finetuning for MolFormer and ChemBERTa2 on Preprty Prediction Tasks
│   ├── Peptides/                      # Finetuning for Peptides tasks (48 NP models)
│   └── Tox21                          # Finetuning for Tox21 tasks (48 NP models)
│
├── /Tokenizers/                       # Tokenizers classes vocab files
│   ├── Vocab Files/                   # Tokenizers .json vocab files
│   └── tokenisers.py                  # Tokenizers classes 
│ 
├── /Generated pseudo NPs/             # Generated NP strings from all 48 models 
│ 
├── /Pretrained NP CLMs/               # Pretrained model checkpoint/weight files (20 out of the 48)
│
├── sam.py                             # Sharpness Aware Minimization 
│
├── README.md                          # Repo structure documentation (this file)
└── LICENSE                            # License information
