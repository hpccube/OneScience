"""Run Chai-1 structure prediction through the OneScience API."""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch

from onescience.models.chai1 import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path, help="Chai-1 FASTA-like input file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trunk-recycles", type=int, default=1)
    parser.add_argument("--num-diffusion-timesteps", type=int, default=20)
    parser.add_argument("--num-diffusion-samples", type=int, default=1)
    parser.add_argument("--num-trunk-samples", type=int, default=1)
    parser.add_argument("--msa-directory", type=Path)
    parser.add_argument("--constraint-path", type=Path)
    parser.add_argument("--template-hits-path", type=Path)
    parser.add_argument("--use-msa-server", action="store_true")
    parser.add_argument("--use-templates-server", action="store_true")
    parser.add_argument("--no-esm-embeddings", action="store_true")
    parser.add_argument("--no-low-memory", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.fasta.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {args.fasta}")
    if not args.model_dir.is_dir():
        raise NotADirectoryError(f"Chai-1 model directory not found: {args.model_dir}")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA-compatible device is unavailable: {args.device}")
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device_index} is unavailable; "
                f"visible device count is {torch.cuda.device_count()}"
            )
    positive_parameters = {
        "--num-trunk-recycles": args.num_trunk_recycles,
        "--num-diffusion-timesteps": args.num_diffusion_timesteps,
        "--num-diffusion-samples": args.num_diffusion_samples,
        "--num-trunk-samples": args.num_trunk_samples,
    }
    for name, value in positive_parameters.items():
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
    if args.msa_directory is not None and args.use_msa_server:
        raise ValueError("--msa-directory and --use-msa-server are mutually exclusive")
    if args.template_hits_path is not None and args.use_templates_server:
        raise ValueError(
            "--template-hits-path and --use-templates-server are mutually exclusive"
        )
    if args.msa_directory is not None and not args.msa_directory.is_dir():
        raise NotADirectoryError(f"MSA directory not found: {args.msa_directory}")
    for name, path in (
        ("constraint", args.constraint_path),
        ("template hits", args.template_hits_path),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{name.title()} file not found: {path}")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    resolved_output = output_dir.expanduser().resolve()
    protected_paths = {
        Path(resolved_output.anchor),
        Path.home().resolve(),
        Path.cwd().resolve(),
    }
    if resolved_output in protected_paths:
        raise ValueError(
            f"Refusing to use protected path as output directory: {resolved_output}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    args.fasta = args.fasta.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    for name in ("msa_directory", "constraint_path", "template_hits_path"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    validate_args(args)
    prepare_output_dir(args.output_dir, args.overwrite)

    candidates = run_inference(
        fasta_file=args.fasta,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        device=args.device,
        seed=args.seed,
        num_trunk_recycles=args.num_trunk_recycles,
        num_diffn_timesteps=args.num_diffusion_timesteps,
        num_diffn_samples=args.num_diffusion_samples,
        num_trunk_samples=args.num_trunk_samples,
        msa_directory=args.msa_directory,
        constraint_path=args.constraint_path,
        template_hits_path=args.template_hits_path,
        use_msa_server=args.use_msa_server,
        use_templates_server=args.use_templates_server,
        use_esm_embeddings=not args.no_esm_embeddings,
        low_memory=not args.no_low_memory,
    ).sorted()

    summary = {
        "input": str(args.fasta.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "device": args.device,
        "candidates": [
            {
                "rank": rank,
                "cif_path": str(cif_path.resolve()),
                "aggregate_score": ranking.aggregate_score.item(),
            }
            for rank, (cif_path, ranking) in enumerate(
                zip(candidates.cif_paths, candidates.ranking_data, strict=True), start=1
            )
        ],
    }
    summary_path = args.output_dir / "ranking.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote ranked prediction summary to %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
