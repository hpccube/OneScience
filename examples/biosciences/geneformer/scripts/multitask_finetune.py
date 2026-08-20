"""Fine-tune Geneformer jointly on multiple cell-classification tasks."""

import argparse
import json
from pathlib import Path

import torch

from onescience.utils.geneformer import MTLClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multitask"))
    parser.add_argument("--task-column", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--freeze-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distributed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Geneformer multi-task fine-tuning requires a CUDA/DTK device")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    results_dir = args.output_dir / "results"
    manual_hyperparameters = {
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.01,
        "weight_decay": args.weight_decay,
        "dropout_rate": args.dropout_rate,
        "lr_scheduler_type": "cosine",
        "use_attention_pooling": False,
        "task_weights": [1.0] * len(args.task_column),
        "max_layers_to_freeze": args.freeze_layers,
    }
    classifier = MTLClassifier(
        task_columns=args.task_column,
        train_path=str(args.train_data),
        val_path=str(args.validation_data),
        pretrained_path=str(args.model_dir),
        model_save_path=str(model_dir),
        results_dir=str(results_dir),
        trials_result_path=str(results_dir / "trials.txt"),
        tensorboard_log_dir=str(args.output_dir / "tensorboard"),
        batch_size=args.batch_size,
        epochs=args.epochs,
        distributed_training=args.distributed,
        use_manual_hyperparameters=True,
        manual_hyperparameters=manual_hyperparameters,
        seed=args.seed,
    )
    classifier.run_manual_tuning()
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved multi-task fine-tuning outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
