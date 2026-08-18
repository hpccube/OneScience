#!/usr/bin/env python3
"""Extract NequIP demo launch metadata and its Hydra training config."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import yaml


META_KEYS = {"name", "launch", "slurm", "env"}


def _assignment(name: str, value) -> None:
    print(f"{name}={shlex.quote(str(value))}")


def _positive_int(value, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _config(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(
        r"\$\{demo_dir:([^}]+)\}",
        lambda match: str(Path(__file__).parent.resolve() / match.group(1)),
        text,
    )
    return yaml.safe_load(text) or {}


def _print_launch(cfg: dict) -> None:
    launch = cfg.get("launch", {}) or {}
    trainer = cfg.get("trainer", {}) or {}
    mode = launch.get("mode", "local")
    if mode not in {"auto", "local", "submit"}:
        raise ValueError("launch.mode must be 'auto', 'local', or 'submit'")

    nodes = _positive_int(launch.get("num_nodes", 1), "launch.num_nodes")
    devices = _positive_int(launch.get("num_gpus", 1), "launch.num_gpus")
    trainer_nodes = _positive_int(trainer.get("num_nodes", 1), "trainer.num_nodes")
    trainer_devices = _positive_int(trainer.get("devices", 1), "trainer.devices")
    if nodes != trainer_nodes:
        raise ValueError("launch.num_nodes must equal trainer.num_nodes")
    if devices != trainer_devices:
        raise ValueError("launch.num_gpus must equal trainer.devices")
    _assignment("RUN_MODE", mode)
    _assignment("NODES", nodes)
    _assignment("GPUS_PER_NODE", devices)
    _assignment("WORLD_SIZE", nodes * devices)


def _print_slurm(cfg: dict) -> None:
    slurm = cfg.get("slurm", {}) or {}
    _assignment("PARTITION", slurm.get("partition", "hx1hdnormal01"))
    _assignment("TIME", slurm.get("time", "01:00:00"))
    _assignment(
        "CPUS_PER_TASK",
        _positive_int(slurm.get("cpus_per_task", 8), "slurm.cpus_per_task"),
    )
    _assignment("NODELIST", slurm.get("nodelist", ""))


def _print_env(cfg: dict) -> None:
    env = cfg.get("env", {}) or {}
    for name, value in env.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid environment variable name: {name}")
        _assignment(f"export {name}", value)


def _print_training_config(cfg: dict) -> None:
    training = {key: value for key, value in cfg.items() if key not in META_KEYS}
    yaml.safe_dump(
        training,
        sys.stdout,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: _parse_config.py <config.yaml> "
            "<name|launch|slurm|env|training-config>"
        )
    cfg = _config(sys.argv[1])
    action = sys.argv[2]
    actions = {
        "name": lambda: print(cfg.get("name", "nequip_run")),
        "launch": lambda: _print_launch(cfg),
        "slurm": lambda: _print_slurm(cfg),
        "env": lambda: _print_env(cfg),
        "training-config": lambda: _print_training_config(cfg),
        "finetune-config": lambda: _print_training_config(cfg),
    }
    try:
        actions[action]()
    except KeyError as error:
        raise SystemExit(f"unknown action: {action}") from error


if __name__ == "__main__":
    main()
