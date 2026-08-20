"""Fine-tune Geneformer for gene classification."""

import argparse
import pickle
from pathlib import Path

import torch

from onescience.utils.geneformer import Classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--gene-class-dict", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/gene_finetune")
    )
    parser.add_argument("--output-prefix", default="tf_dosage_sensitivity")
    parser.add_argument("--model-version", choices=("V1", "V2"), default="V1")
    parser.add_argument("--max-cells", type=int, default=10_000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--freeze-layers", type=int, default=4)
    parser.add_argument(
        "--cross-validation-splits", type=int, choices=(1, 5), default=5
    )
    parser.add_argument("--forward-batch-size", type=int, default=200)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gene-balance", action="store_true")
    parser.add_argument("--train-all-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--hyperparameter-trials",
        type=int,
        default=0,
        help="Run the official Ray/HyperOpt search for this many trials.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_all_data and args.hyperparameter_trials:
        raise ValueError("--hyperparameter-trials cannot be used with --train-all-data")
    with args.gene_class_dict.open("rb") as handle:
        gene_class_dict = pickle.load(handle)
    if not isinstance(gene_class_dict, dict) or len(gene_class_dict) < 2:
        raise ValueError("gene-class-dict must map at least two class names to gene lists")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = {
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "seed": args.seed,
        "report_to": "none",
    }
    if args.max_steps is not None:
        training_args["max_steps"] = args.max_steps

    classifier = Classifier(
        classifier="gene",
        gene_class_dict=gene_class_dict,
        max_ncells=args.max_cells,
        training_args=training_args,
        freeze_layers=args.freeze_layers,
        num_crossval_splits=(
            0 if args.train_all_data else args.cross_validation_splits
        ),
        forward_batch_size=args.forward_batch_size,
        model_version=args.model_version,
        nproc=args.nproc,
        ngpu=max(torch.cuda.device_count(), 1),
    )
    classifier.prepare_data(
        input_data_file=args.data_file,
        output_directory=args.output_dir,
        output_prefix=args.output_prefix,
    )
    if args.prepare_only:
        print(f"Prepared gene-labeled dataset in {args.output_dir}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Geneformer fine-tuning requires a visible CUDA/DTK device")

    common_args = {
        "model_directory": str(args.model_dir),
        "prepared_input_data_file": str(
            args.output_dir / f"{args.output_prefix}_labeled.dataset"
        ),
        "id_class_dict_file": str(
            args.output_dir / f"{args.output_prefix}_id_class_dict.pkl"
        ),
        "output_directory": str(args.output_dir),
        "output_prefix": args.output_prefix,
        "gene_balance": args.gene_balance,
    }
    if args.train_all_data:
        classifier.train_all_data(**common_args)
    else:
        classifier.validate(
            **common_args,
            n_hyperopt_trials=args.hyperparameter_trials,
        )
    print(f"Saved gene-classification outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
