"""Evaluate a local DNABERT-2 classifier or PEFT adapter."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from onescience.datapipes.dnabert2 import SequenceBatchCollator, SequenceClassificationDataset
from onescience.metrics.dnabert2 import classification_metrics
from onescience.models.dnabert2 import load_tokenizer, load_trained_classifier
from onescience.models.dnabert2 import hf_architecture


DEFAULT_ARCHITECTURE_DIR = Path(hf_architecture.__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path)
    parser.add_argument(
        "--architecture-dir",
        type=Path,
        default=DEFAULT_ARCHITECTURE_DIR,
        help="DNABERT-2 architecture bundle (defaults to the copy installed with OneScience)",
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-max-length", type=int)
    parser.add_argument("--kmer", type=int)
    parser.add_argument("--num-labels", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.checkpoint_dir / "dnabert2_training.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    num_labels = args.num_labels or metadata.get("num_labels")
    model_max_length = args.model_max_length or metadata.get("model_max_length", 512)
    kmer = args.kmer if args.kmer is not None else metadata.get("kmer", -1)
    base_model_dir = args.base_model_dir or metadata.get("base_model_dir")
    dataset = SequenceClassificationDataset(args.data_file, num_labels=num_labels)
    tokenizer_dir = base_model_dir or args.checkpoint_dir
    tokenizer = load_tokenizer(tokenizer_dir, model_max_length=model_max_length)
    model = load_trained_classifier(
        args.checkpoint_dir,
        base_model_dir=base_model_dir,
        architecture_dir=args.architecture_dir,
        num_labels=num_labels or dataset.num_labels,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=SequenceBatchCollator(
            tokenizer,
            max_length=model_max_length,
            kmer=kmer,
        ),
    )
    predictions = []
    labels = []
    with torch.inference_mode():
        for batch in loader:
            batch_labels = batch.pop("labels")
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch_labels.tolist())

    metrics = classification_metrics(np.asarray(predictions), np.asarray(labels))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "prediction"])
        writer.writerows((index, label, prediction) for index, (label, prediction) in enumerate(zip(labels, predictions)))


if __name__ == "__main__":
    main()
