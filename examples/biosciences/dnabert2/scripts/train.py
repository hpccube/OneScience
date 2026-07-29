"""Full-parameter or LoRA supervised training for a local DNABERT-2 model."""

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml

from onescience.datapipes.dnabert2 import SequenceBatchCollator, SequenceClassificationDataset
from onescience.metrics.dnabert2 import classification_metrics
from onescience.models.dnabert2 import load_sequence_classifier, load_tokenizer

EXAMPLE_DIR = Path(__file__).resolve().parent.parent


def parse_args(default_lora: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXAMPLE_DIR / "configs/train.yaml")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--use-lora", action="store_true", default=default_lora)
    parser.add_argument("--model-max-length", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--per-device-eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def _load_settings(args: argparse.Namespace) -> dict[str, Any]:
    with args.config.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    training = settings["training"]
    for name in (
        "model_max_length",
        "learning_rate",
        "num_train_epochs",
        "max_steps",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "save_steps",
        "eval_steps",
        "warmup_steps",
        "seed",
        "fp16",
        "bf16",
    ):
        value = getattr(args, name)
        if value is not None:
            training[name] = value
    return settings


def _training_arguments(settings: dict[str, Any], output_dir: Path):
    values = settings["training"]
    kwargs = {
        "output_dir": str(output_dir),
        "run_name": values["run_name"],
        "optim": values["optim"],
        "learning_rate": values["learning_rate"],
        "weight_decay": values["weight_decay"],
        "num_train_epochs": values["num_train_epochs"],
        "max_steps": values.get("max_steps", -1),
        "per_device_train_batch_size": values["per_device_train_batch_size"],
        "per_device_eval_batch_size": values["per_device_eval_batch_size"],
        "gradient_accumulation_steps": values["gradient_accumulation_steps"],
        "warmup_steps": values["warmup_steps"],
        "logging_steps": values["logging_steps"],
        "save_steps": values["save_steps"],
        "eval_steps": values["eval_steps"],
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "load_best_model_at_end": True,
        "save_total_limit": values["save_total_limit"],
        "fp16": bool(values["fp16"]),
        "bf16": bool(values.get("bf16", False)),
        "seed": values["seed"],
        "report_to": values.get("report_to", []),
        "remove_unused_columns": False,
        "dataloader_pin_memory": values.get("dataloader_pin_memory", False),
        "ddp_find_unused_parameters": values.get("find_unused_parameters", False),
    }
    parameters = inspect.signature(transformers.TrainingArguments.__init__).parameters
    strategy_name = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    kwargs[strategy_name] = "steps"
    return transformers.TrainingArguments(**kwargs)


def _preprocess_logits(logits, _labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def _compute_metrics(eval_prediction) -> dict[str, float]:
    return classification_metrics(
        np.asarray(eval_prediction.predictions),
        np.asarray(eval_prediction.label_ids),
    )


def main(default_lora: bool = False) -> None:
    args = parse_args(default_lora=default_lora)
    settings = _load_settings(args)
    values = settings["training"]
    train_dataset = SequenceClassificationDataset(args.data_dir / "train.csv")
    validation_dataset = SequenceClassificationDataset(
        args.data_dir / "dev.csv",
        num_labels=train_dataset.num_labels,
    )
    test_dataset = SequenceClassificationDataset(
        args.data_dir / "test.csv",
        num_labels=train_dataset.num_labels,
    )
    tokenizer = load_tokenizer(args.model_dir, model_max_length=values["model_max_length"])
    model = load_sequence_classifier(args.model_dir, num_labels=train_dataset.num_labels)

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora = settings["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora["r"],
                lora_alpha=lora["alpha"],
                target_modules=lora["target_modules"],
                lora_dropout=lora["dropout"],
                bias="none",
                task_type="SEQ_CLS",
                inference_mode=False,
            ),
        )
        model.print_trainable_parameters()

    collator = SequenceBatchCollator(
        tokenizer,
        max_length=values["model_max_length"],
        kmer=settings["data"].get("kmer", -1),
    )
    trainer_kwargs = {
        "model": model,
        "args": _training_arguments(settings, args.output_dir),
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": collator,
        "preprocess_logits_for_metrics": _preprocess_logits,
        "compute_metrics": _compute_metrics,
    }
    trainer_parameters = inspect.signature(transformers.Trainer.__init__).parameters
    trainer_kwargs[
        "processing_class" if "processing_class" in trainer_parameters else "tokenizer"
    ] = tokenizer
    trainer = transformers.Trainer(**trainer_kwargs)
    trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "dnabert2_training.json").write_text(
        json.dumps(
            {
                "base_model_dir": str(args.model_dir.expanduser().resolve()),
                "num_labels": train_dataset.num_labels,
                "model_max_length": values["model_max_length"],
                "kmer": settings["data"].get("kmer", -1),
                "use_lora": args.use_lora,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "eval_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
