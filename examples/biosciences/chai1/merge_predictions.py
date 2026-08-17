"""Merge candidate files produced by independent Chai-1 GPU workers."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("worker_dirs", type=Path, nargs="+")
    return parser.parse_args()


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


def score_path_for(candidate_path: Path) -> Path:
    if not candidate_path.name.startswith("pred.") or candidate_path.suffix != ".cif":
        raise ValueError(f"Unexpected candidate filename: {candidate_path}")
    return candidate_path.with_name(
        candidate_path.name.replace("pred.", "scores.", 1).replace(".cif", ".npz")
    )


def merge_predictions(output_dir: Path, worker_dirs: list[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_candidates: list[dict[str, object]] = []
    copied_msa_plot = False
    candidate_scores: list[tuple[Path, Path]] = []
    for worker_dir in worker_dirs:
        summary_path = worker_dir / "ranking.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Worker summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        candidates = summary.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Worker summary has no candidates: {summary_path}")
        for candidate in candidates:
            source_cif = Path(candidate["cif_path"])
            if not source_cif.is_file():
                raise FileNotFoundError(f"Worker candidate not found: {source_cif}")
            index = len(merged_candidates)
            target_cif = output_dir / f"pred.model_idx_{index}.cif"
            shutil.copy2(source_cif, target_cif)

            source_scores = score_path_for(source_cif)
            if not source_scores.is_file():
                raise FileNotFoundError(f"Worker scores not found: {source_scores}")
            candidate_scores.append((source_scores, score_path_for(target_cif)))

            merged_candidate = dict(candidate)
            merged_candidate["rank"] = index + 1
            merged_candidate["cif_path"] = str(target_cif.resolve())
            merged_candidates.append(merged_candidate)

        if not copied_msa_plot:
            source_plot = worker_dir / "msa_depth.pdf"
            if source_plot.is_file():
                shutil.copy2(source_plot, output_dir / source_plot.name)
                copied_msa_plot = True

    merged_candidates.sort(
        key=lambda item: float(item.get("aggregate_score", float("-inf"))),
        reverse=True,
    )
    for rank, candidate in enumerate(merged_candidates, start=1):
        candidate["rank"] = rank

    # Candidate filenames are assigned before sorting so downstream files retain
    # their stable model_idx relationship with the merged summary.
    for source_scores, target_scores in candidate_scores:
        shutil.copy2(source_scores, target_scores)

    first_summary = json.loads(
        (worker_dirs[0] / "ranking.json").read_text(encoding="utf-8")
    )
    first_summary["device"] = "multi"
    first_summary["candidates"] = merged_candidates
    (output_dir / "ranking.json").write_text(
        json.dumps(first_summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.worker_dirs = [path.expanduser().resolve() for path in args.worker_dirs]
    prepare_output_dir(args.output_dir, args.overwrite)
    merge_predictions(args.output_dir, args.worker_dirs)


if __name__ == "__main__":
    main()
