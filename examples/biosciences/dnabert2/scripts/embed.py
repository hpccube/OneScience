"""Generate pooled DNABERT-2 embeddings from local sequence input."""

import argparse
from pathlib import Path

import numpy as np
import yaml

from onescience.datapipes.dnabert2 import read_sequence_records
from onescience.models.dnabert2 import DNABert2Encoder

EXAMPLE_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXAMPLE_DIR / "configs/inference.yaml",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pooling", choices=["mean", "max", "cls"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--model-max-length", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    pooling = args.pooling or settings["pooling"]
    batch_size = args.batch_size or settings["batch_size"]
    model_max_length = args.model_max_length or settings["model_max_length"]
    records = read_sequence_records(args.input)
    encoder = DNABert2Encoder(
        args.model_dir,
        device=args.device,
        model_max_length=model_max_length,
    )
    embeddings = encoder.embed(
        [record.sequence for record in records],
        batch_size=batch_size,
        pooling=pooling,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        identifiers=np.asarray([record.identifier for record in records]),
        embeddings=embeddings,
    )


if __name__ == "__main__":
    main()
