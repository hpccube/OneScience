"""Fine-tune Geneformer for cell-state classification."""

import argparse
from pathlib import Path

import torch

from onescience.utils.geneformer import Classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/finetune"))
    parser.add_argument("--output-prefix", default="disease_classifier")
    parser.add_argument("--state-column", default="disease")
    parser.add_argument(
        "--state",
        action="append",
        default=[],
        help="Class value to include; repeat the option. The default uses all classes.",
    )
    parser.add_argument("--filter-column")
    parser.add_argument("--filter-value", action="append", default=[])
    parser.add_argument("--model-version", choices=("V1", "V2"), default="V1")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--freeze-layers", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--forward-batch-size", type=int, default=100)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hyperparameter-trials",
        type=int,
        default=0,
        help="Run the official Ray/Optuna hyperparameter search for this many trials.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validation_fraction <= 0 or args.test_fraction <= 0:
        raise ValueError("validation-fraction and test-fraction must both be positive")
    train_fraction = 1.0 - args.validation_fraction - args.test_fraction
    if train_fraction <= 0:
        raise ValueError("validation-fraction + test-fraction must be less than 1")
    if args.filter_column and not args.filter_value:
        raise ValueError("--filter-column requires at least one --filter-value")
    if args.filter_value and not args.filter_column:
        raise ValueError("--filter-value requires --filter-column")

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
        classifier="cell",
        cell_state_dict={
            "state_key": args.state_column,
            "states": args.state or "all",
        },
        filter_data=(
            {args.filter_column: args.filter_value} if args.filter_column else None
        ),
        max_ncells=args.max_cells,
        training_args=training_args,
        freeze_layers=args.freeze_layers,
        num_crossval_splits=1,
        split_sizes={
            "train": train_fraction,
            "valid": args.validation_fraction,
            "test": args.test_fraction,
        },
        forward_batch_size=args.forward_batch_size,
        model_version=args.model_version,
        nproc=args.nproc,
        ngpu=max(torch.cuda.device_count(), 1),
    )
    classifier.prepare_data(
        input_data_file=args.data_file,
        output_directory=args.output_dir,
        output_prefix=args.output_prefix,
        test_size=args.test_fraction,
    )
    if args.prepare_only:
        print(f"Prepared labeled datasets in {args.output_dir}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Geneformer fine-tuning requires a visible CUDA/DTK device")

    classifier.validate(
        model_directory=str(args.model_dir),
        prepared_input_data_file=str(
            args.output_dir / f"{args.output_prefix}_labeled_train.dataset"
        ),
        id_class_dict_file=str(
            args.output_dir / f"{args.output_prefix}_id_class_dict.pkl"
        ),
        output_directory=str(args.output_dir),
        output_prefix=args.output_prefix,
        n_hyperopt_trials=args.hyperparameter_trials,
    )
    print(f"Saved fine-tuning outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
