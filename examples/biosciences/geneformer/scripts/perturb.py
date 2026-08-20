"""Run Geneformer zero-shot in-silico gene perturbation."""

import argparse
import json
from pathlib import Path

import torch

from onescience.utils.geneformer import InSilicoPerturber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/perturb"))
    parser.add_argument("--output-prefix", default="geneformer_perturb")
    parser.add_argument("--model-version", choices=("V1", "V2"), default="V1")
    parser.add_argument(
        "--model-type",
        choices=("Pretrained", "GeneClassifier", "CellClassifier"),
        default="Pretrained",
    )
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--perturb-type", choices=("delete", "overexpress"), default="delete")
    parser.add_argument(
        "--gene",
        action="append",
        default=[],
        help="Ensembl ID to perturb; repeat to perturb a group. Default: every detected gene.",
    )
    parser.add_argument("--emb-mode", choices=("cls", "cell", "cls_and_gene", "cell_and_gene"), default="cell")
    parser.add_argument("--max-cells", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--filter-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Geneformer perturbation requires a visible CUDA/DTK device")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    perturber = InSilicoPerturber(
        perturb_type=args.perturb_type,
        genes_to_perturb=args.gene or "all",
        model_type=args.model_type,
        num_classes=args.num_classes,
        emb_mode=args.emb_mode,
        filter_data=json.loads(args.filter_json) if args.filter_json else None,
        max_ncells=args.max_cells,
        forward_batch_size=args.batch_size,
        nproc=args.nproc,
        model_version=args.model_version,
    )
    perturber.perturb_data(
        model_directory=args.model_dir,
        input_data_file=args.data_file,
        output_directory=args.output_dir,
        output_prefix=args.output_prefix,
    )
    print(f"Saved perturbation batches under {args.output_dir}")


if __name__ == "__main__":
    main()
