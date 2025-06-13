# main.py
import os
import wandb
import argparse
from mol_generation import run_generation
from hpsearch import run_hpsearch
from pretraining import run_pretraining
from finetuning import run_finetuning
from ChemBERTa2_finetuning import run_chemberta
from MolFormer_finetuning import run_molformer
from MolFormer_Finetuning_on_1M_NPs import run_molformer_finetuning_on_1M_NPs

def main():
    parser = argparse.ArgumentParser(
        description="Run molecule generation, HP search, pretraining, or finetuning"
    )

    # ─── Common arguments ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--task",
        choices=["generate", "hpsearch", "pretrain", "finetune", "chemberta", "molformer_1M_NPs", "molformer"],
        required=True,
        help="Which sub‐command to run"
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=None,
        help="(only for pretraining the 48 models and finetuning MolFormer on 1M NP) Your Wandb API key"
    )

    # ─── Molecule Generation arguments ───────────────────────────────────────────────
    parser.add_argument(
        "--num_mols",
        type=int,
        default=32,
        help="(generate) Number of molecules to sample"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="(generate) Sampling temperature"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="(generate) Max token length"
    )
    
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=None,
        help="(generate) List of model names to use"
    )

    # ─── Hyperparameter search arguments ─────────────────────────────────────────────────
    parser.add_argument(
        "--hp_model",
        choices=["GPT", "Mamba1", "Mamba2"],
        default="GPT",
        help="(hpsearch) Model type"
    )
    parser.add_argument(
        "--hp_tokenizer",
        choices=["Char", "AIS", "BPE", "NPBPE60", "NPBPE100", "NPBPE1000", "NPBPE7924", "NPBPE30k"],
        default="AIS",
        help="(hpsearch) Tokenizer"
    )
    parser.add_argument(
        "--hp_split",
        choices=["random", "scaffold"],
        default="random",
        help="(hpsearch) Data split"
    )

    # ─── Pre-training arguments ──────────────────────────────────────────────
    parser.add_argument(
        "--pt_model",
        choices=["GPT", "Mamba1", "Mamba2"],
        default="GPT",
        help="(pretrain) Model type"
    )
    parser.add_argument(
        "--pt_tokenizer",
        choices=["Char", "AIS", "BPE", "NPBPE60", "NPBPE100", "NPBPE1000", "NPBPE7924", "NPBPE30k"],
        default="NPBPE100",
        help="(pretrain) Tokenizer"
    )
    parser.add_argument(
        "--pt_split",
        choices=["random", "scaffold"],
        default="random",
        help="(pretrain) Data split"
    )
    parser.add_argument(
        "--pt_n_embd",
        type=int,
        default=256,
        help="(pretrain) Hidden dimension"
    )
    parser.add_argument(
        "--pt_n_layer",
        type=int,
        default=8,
        help="(pretrain) Number of layers"
    )
    parser.add_argument(
        "--pt_lr",
        type=float,
        default=1e-4,
        help="(pretrain) Learning rate"
    )
    
    def none_or_int(val):
        return None if val.lower() == "none" else int(val)

    parser.add_argument(
        "--pt_n_head",
        type=none_or_int,
        default=None,
        help="(for GPT) Number of heads; use 'None' for non-GPT models"
    )
    
    # ─── Fine-tuning arguments ──────────────────────────────────────────────
    parser.add_argument(
        "--sub_task",
        choices=["anti_cancer", "peptides", "tastes"],
        default="peptides",
        help="(finetune) Which downstream task to run"
    )

    parser.add_argument(
        "--model_split",
        choices=["rds", "sfs"],
        default="rds",
        help="(finetune) Which version of model to use (rds or sfs)"
    )

    parser.add_argument(
        "--data_split",
        choices=["rd", "sf"],
        default="rd",
        help="(finetune) Which version of data to use (rd or sf)"
    )
    
    # ─── Fine-tuning ChemBERTa-2 arguments ──────────────────────────────────────────────
    parser.add_argument(
        "--chemberta_model_type",
        choices=["mlm", "mtr", "mlm-finetuned"],
        default="mlm",
        help="(chemberta) Model variant to use"
    )
    
    # ─── Fine-tuning MolFormer arguments ──────────────────────────────────────────────
    parser.add_argument(
        "--molformer_variant",
        choices=["molformer", "molformer-finetuned"],
        default="molformer",
        help="(molformer) Use original or fine-tuned MoLFormer"
    )

    args = parser.parse_args()

    # ─── Log into Wandb for pretraining ────────────────────────────────────
    if args.task == "pretrain":
        if not args.wandb_key:
            raise RuntimeError(
                "Task 'pretrain' requires --wandb_key. "
                "Run with e.g.:\n"
                "  python main.py --task pretrain --wandb_key YOUR_KEY  "
            )
        wandb.login(key=args.wandb_key)

    # ─── Dispatch by task ─────────────────────────────────────────────────────────────
    if args.task == "generate":
        shared_config = {
            "num_mols": args.num_mols,
            "temperature": args.temperature,
            "max_length": args.max_length,
        }

        model_names = args.model_names or ["rozariwang/GPT-NPBPE100-rds"]

        print(f"→ Running generation with num_mols={args.num_mols}, temperature={args.temperature}")
        for model_name in model_names:
            model_id = os.path.basename(model_name)
            cfg = {
                "model_name": model_name,
                "outfile": f"{model_id}_generated.csv",
                **shared_config
            }
            print(f"   • Model: {model_id}")
            run_generation(cfg)

    elif args.task == "hpsearch":
        cfg = {
            "model": args.hp_model,
            "tokenizer": args.hp_tokenizer,
            "split": args.hp_split
        }
        print(
            f"→ Running HP search | "
            f"Model={args.hp_model} | Tokenizer={args.hp_tokenizer} | Split={args.hp_split}"
        )
        run_hpsearch(cfg)

    elif args.task == "pretrain":
        cfg = {
            "model": args.pt_model,
            "tokenizer": args.pt_tokenizer,
            "split": args.pt_split,
            "n_embd": args.pt_n_embd,
            "n_layer": args.pt_n_layer,
            "lr": args.pt_lr
        }
        if args.pt_model.lower() == "gpt":
            cfg["n_head"] = 8

        msg = (
            f"→ Running pretraining | "
            f"Model={args.pt_model} | Tokenizer={args.pt_tokenizer} | Split={args.pt_split} | "
            f"n_embd={args.pt_n_embd} | n_layer={args.pt_n_layer} | lr={args.pt_lr}"
        )
        if args.pt_model.lower() == "gpt":
            msg += f" | n_head={args.pt_n_head}"
        print(msg)

        if args.pt_model.lower() == "gpt" and args.pt_n_head is None:
            parser.error("GPT model requires --pt_n_head to be set.")
            
        run_pretraining(cfg)
        
    elif args.task == "finetune":
        cfg = {
            "data_split": args.data_split,      # for dataset selection
            "model_split": args.model_split,    # pass model suffix directly
            "sub_task": args.sub_task           # single task or 'all'
        }
        print(f"→ Running fine‐tuning  {args.sub_task}")
        run_finetuning(cfg)   
        
    elif args.task == "chemberta":
        cfg = {
            "model_type": args.chemberta_model_type,
            "sub_task": args.sub_task,
            "data_split": args.data_split
        }
        print(
            f"→ Running ChemBERTa fine‐tuning | "
            f"Model={args.chemberta_model_type} | Sub-task={args.sub_task} | Data split={args.data_split}"
        )
        run_chemberta(cfg)
        
    elif args.task == "molformer_1M_NPs":
        if not args.wandb_key:
            raise RuntimeError("MolFormer fine-tuning requires --wandb_key.")
        print("→ Running MolFormer fine-tuning on 1M NPs...")
        run_molformer_finetuning_on_1M_NPs(wandb_key=args.wandb_key)
        
    elif args.task == "molformer":
        cfg = {
            "model_type": args.molformer_variant,
            "sub_task": args.sub_task,
            "data_split": args.data_split
        }
        print(
            f"→ Running MolFormer fine‐tuning | "
            f"Model={args.molformer_variant} | Sub-task={args.sub_task} | Data split={args.data_split}"
        )
        run_molformer(cfg)


if __name__ == "__main__":
    main()
