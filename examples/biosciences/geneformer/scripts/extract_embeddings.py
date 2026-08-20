"""Extract cell or gene embeddings from a tokenized Geneformer dataset."""

import argparse
import json
from pathlib import Path

import torch

from onescience.utils.geneformer import EmbExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/embeddings"))
    parser.add_argument("--output-prefix", default="geneformer_embeddings")
    parser.add_argument("--model-version", choices=("V1", "V2"), default="V1")
    parser.add_argument(
        "--model-type",
        choices=("Pretrained", "GeneClassifier", "CellClassifier"),
        default="CellClassifier",
    )
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--emb-mode", choices=("cls", "cell", "gene"), default="cell")
    parser.add_argument("--emb-layer", type=int, choices=(-1, 0), default=0)
    parser.add_argument("--max-cells", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument(
        "--filter-json",
        help='Dataset filter as JSON, for example {"cell_type":["Cardiomyocyte1"]}.',
    )
    parser.add_argument("--save-tensor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Geneformer embedding extraction requires a visible CUDA/DTK device")
    filter_data = json.loads(args.filter_json) if args.filter_json else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extractor = EmbExtractor(
        model_type=args.model_type,
        num_classes=args.num_classes if args.model_type != "Pretrained" else 0,
        emb_mode=args.emb_mode,
        filter_data=filter_data,
        max_ncells=args.max_cells,
        emb_layer=args.emb_layer,
        emb_label=args.label or None,
        forward_batch_size=args.batch_size,
        model_version=args.model_version,
        nproc=args.nproc,
    )
    extractor.extract_embs(
        model_directory=args.model_dir,
        input_data_file=args.data_file,
        output_directory=args.output_dir,
        output_prefix=args.output_prefix,
        output_torch_embs=args.save_tensor,
    )
    print(f"Saved embeddings to {args.output_dir / (args.output_prefix + '.csv')}")


if __name__ == "__main__":
    main()
