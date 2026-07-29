"""Run OpenStructure evaluations for Boltz benchmark predictions."""

import argparse
import concurrent.futures
import re
import subprocess
from pathlib import Path

from tqdm import tqdm


DEFAULT_IMAGE = "openstructure-0.2.8"

OST_COMPARE_STRUCTURE = r"""
#!/bin/bash
# https://openstructure.org/docs/2.7/actions/#ost-compare-structures

IMAGE_NAME={image}

command="compare-structures \
-m {model_file} \
-r {reference_file} \
--fault-tolerant \
--min-pep-length 4 \
--min-nuc-length 4 \
-o {output_path} \
--lddt --bb-lddt --qs-score --dockq \
--ics --ips --rigid-scores --patch-scores --tm-score"

docker run -u $(id -u):$(id -g) --rm --volume {mount}:{mount} $IMAGE_NAME $command
"""


OST_COMPARE_LIGAND = r"""
#!/bin/bash
# https://openstructure.org/docs/2.7/actions/#ost-compare-structures

IMAGE_NAME={image}

command="compare-ligand-structures \
-m {model_file} \
-r {reference_file} \
--fault-tolerant \
--lddt-pli --rmsd \
--substructure-match \
-o {output_path}"

docker run -u $(id -u):$(id -g) --rm --volume {mount}:{mount} $IMAGE_NAME $command
"""


def evaluate_structure(
    name: str,
    pred: Path,
    reference: Path,
    outdir: Path,
    mount: str,
    executable: str = "/bin/bash",
    image: str = DEFAULT_IMAGE,
) -> None:
    """Evaluate one predicted structure against its reference."""
    pred = Path(pred)
    reference = Path(reference)
    outdir = Path(outdir)

    if not pred.is_file():
        raise FileNotFoundError(f"Prediction file not found: {pred}")
    if not reference.is_file():
        raise FileNotFoundError(f"Reference file not found: {reference}")

    polymer_output = outdir / f"{name}.json"
    if polymer_output.exists():
        print(  # noqa: T201
            f"Skipping recomputation of {name} as protein json file already exists",
            flush=True,
        )
    else:
        print(f"Evaluating polymer metrics for {name}", flush=True)  # noqa: T201
        subprocess.run(
            OST_COMPARE_STRUCTURE.format(
                model_file=str(pred),
                reference_file=str(reference),
                output_path=str(polymer_output),
                mount=mount,
                image=image,
            ),
            shell=True,  # noqa: S602
            check=True,
            executable=executable,
        )

    ligand_output = outdir / f"{name}_ligand.json"
    if ligand_output.exists():
        print(  # noqa: T201
            f"Skipping recomputation of {name} as ligand json file already exists",
            flush=True,
        )
    else:
        print(f"Evaluating ligand metrics for {name}", flush=True)  # noqa: T201
        subprocess.run(
            OST_COMPARE_LIGAND.format(
                model_file=str(pred),
                reference_file=str(reference),
                output_path=str(ligand_output),
                mount=mount,
                image=image,
            ),
            shell=True,  # noqa: S602
            check=True,
            executable=executable,
        )


def _prediction_path(
    folder: Path,
    name: str,
    model_id: int,
    prediction_format: str,
    testset: str,
) -> Path:
    if prediction_format == "af3":
        return folder / f"seed-1_sample-{model_id}" / "model.cif"
    if prediction_format == "chai":
        return folder / f"pred.model_idx_{model_id}.cif"

    name_file = (
        f"{name[0].upper()}{name[1:]}" if testset == "casp" else name.lower()
    )
    expected = folder / f"{name_file}_model_{model_id}.cif"
    if expected.is_file():
        return expected

    # Preserve compatibility with prediction folders whose on-disk case differs
    # from the benchmark reference identifier.
    matches = sorted(folder.glob(f"*_model_{model_id}.cif"))
    return matches[0] if len(matches) == 1 else expected


def _discover_model_ids(folder: Path, prediction_format: str) -> list[int]:
    if prediction_format == "af3":
        pattern = re.compile(r"^seed-1_sample-(\d+)$")
        candidates = (
            path
            for path in folder.iterdir()
            if path.is_dir() and (path / "model.cif").is_file()
        )
    elif prediction_format == "chai":
        pattern = re.compile(r"^pred\.model_idx_(\d+)\.cif$")
        candidates = (path for path in folder.iterdir() if path.is_file())
    else:
        pattern = re.compile(r"^.+_model_(\d+)\.cif$")
        candidates = (path for path in folder.iterdir() if path.is_file())

    model_ids = []
    for path in candidates:
        match = pattern.match(path.name)
        if match:
            model_ids.append(int(match.group(1)))
    return sorted(set(model_ids))


def _reference_path(reference_dir: Path, name: str, testset: str) -> Path:
    if testset == "casp":
        return reference_dir / f"{name[0].upper()}{name[1:]}.cif"
    return reference_dir / f"{name.lower()}.cif.gz"


def main(args):
    if not args.data.is_dir():
        raise NotADirectoryError(f"Prediction directory not found: {args.data}")
    if not args.pdb.is_dir():
        raise NotADirectoryError(f"Reference directory not found: {args.pdb}")
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")

    files = sorted(path for path in args.data.iterdir() if path.is_dir())
    if not files:
        raise RuntimeError(f"No prediction target directories found in {args.data}")
    names = {path.name.lower(): path for path in files}

    args.outdir.mkdir(parents=True, exist_ok=True)

    tasks = []
    missing = []
    for name, folder in names.items():
        model_ids = (
            list(range(args.num_samples))
            if args.num_samples is not None
            else _discover_model_ids(folder, args.format)
        )
        if not model_ids:
            missing.append(f"{name}: no {args.format} prediction models found")
            continue

        reference = _reference_path(args.pdb, name, args.testset)
        if not reference.is_file():
            missing.append(f"{name}: missing reference {reference}")
            continue

        for model_id in model_ids:
            pred = _prediction_path(
                folder,
                name,
                model_id,
                args.format,
                args.testset,
            )
            if not pred.is_file():
                missing.append(f"{name}: missing model {model_id}: {pred}")
                continue
            tasks.append((name, model_id, pred, reference))

    if missing:
        details = "\n".join(f"  - {item}" for item in missing[:20])
        suffix = "\n  - ..." if len(missing) > 20 else ""
        raise FileNotFoundError(
            "Evaluation inputs are incomplete:\n" + details + suffix
        )
    if not tasks:
        raise RuntimeError("No prediction/reference pairs are available for evaluation")

    print(  # noqa: T201
        f"Found {len(names)} targets and {len(tasks)} prediction models; "
        f"writing evaluations to {args.outdir}",
        flush=True,
    )

    first_name, first_model_id, first_pred, first_reference = tasks[0]
    evaluate_structure(
        name=f"{first_name}_model_{first_model_id}",
        pred=first_pred,
        reference=first_reference,
        outdir=args.outdir,
        mount=args.mount,
        image=args.image,
        executable=args.executable,
    )

    remaining = tasks[1:]
    with concurrent.futures.ThreadPoolExecutor(args.max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_structure,
                name=f"{name}_model_{model_id}",
                pred=pred,
                reference=reference,
                outdir=args.outdir,
                mount=args.mount,
                image=args.image,
                executable=args.executable,
            )
            for name, model_id, pred, reference in remaining
        ]
        with tqdm(
            total=len(futures),
            desc="Evaluating structures",
            unit="model",
        ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)

    print(f"Evaluation complete: {args.outdir}", flush=True)  # noqa: T201


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, nargs="?")
    parser.add_argument("pdb", type=Path, nargs="?")
    parser.add_argument("outdir", type=Path, nargs="?")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--output", "--outdir", dest="output", type=Path)
    parser.add_argument("--format", choices=("af3", "chai", "boltz"), default="af3")
    parser.add_argument("--testset", choices=("casp", "test"), default="casp")
    parser.add_argument("--mount", type=str, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--executable", default="/bin/bash")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    args.data = args.predictions or args.data
    args.pdb = args.references or args.pdb
    args.outdir = args.output or args.outdir
    missing = [
        name
        for name, value in (
            ("predictions", args.data),
            ("references", args.pdb),
            ("output", args.outdir),
        )
        if value is None
    ]
    if missing:
        parser.error("missing required paths: " + ", ".join(missing))
    return args


if __name__ == "__main__":
    main(parse_args())
