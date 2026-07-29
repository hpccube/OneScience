"""Run the official DNABERT-2 GUE supervised task matrix."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=EXAMPLE_DIR / "configs/gue.yaml")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=EXAMPLE_DIR / "configs/train.yaml",
    )
    parser.add_argument("--group", action="append", help="Run only the named group")
    parser.add_argument("--task", action="append", help="Run only the named task")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.matrix.open(encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    selected_groups = set(args.group or matrix["groups"].keys())
    selected_tasks = set(args.task or [])

    commands = []
    for group_name, group in matrix["groups"].items():
        if group_name not in selected_groups:
            continue
        for task in map(str, group["tasks"]):
            if selected_tasks and task not in selected_tasks:
                continue
            data_dir = args.dataset_root / group["dataset_parent"] / task
            output_dir = args.output_root / group_name / task
            command = [
                sys.executable,
                str(EXAMPLE_DIR / "scripts/train.py"),
                "--config",
                str(args.train_config),
                "--model-dir",
                str(args.model_dir),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--model-max-length",
                str(group["model_max_length"]),
                "--learning-rate",
                str(matrix["learning_rate"]),
                "--num-train-epochs",
                str(group["num_train_epochs"]),
                "--per-device-train-batch-size",
                str(group["train_batch_size"]),
                "--per-device-eval-batch-size",
                str(group["eval_batch_size"]),
                "--save-steps",
                str(group["save_steps"]),
                "--eval-steps",
                str(group["eval_steps"]),
                "--warmup-steps",
                str(group["warmup_steps"]),
                "--seed",
                str(matrix["seed"]),
                "--fp16",
            ]
            if group.get("max_steps") is not None:
                command.extend(["--max-steps", str(group["max_steps"])])
            commands.append(command)

    if not commands:
        raise ValueError("No GUE tasks matched the selected group/task filters")
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
