#!/usr/bin/env python3
"""Extract Equiformer V3 demo metadata and its fine-tuning configuration."""

from __future__ import annotations

import shlex
import sys

import yaml


META_KEYS = {"name", "description", "launch", "slurm", "nccl"}


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
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _launch(cfg: dict) -> dict:
    return cfg.get("launch", {}) or {}


def _slurm(cfg: dict) -> dict:
    return cfg.get("slurm", {}) or {}


def _print_launch(cfg: dict) -> None:
    launch = _launch(cfg)
    mode = launch.get("mode", "local")
    if mode not in {"local", "submit"}:
        raise ValueError("launch.mode must be 'local' or 'submit'")
    _assignment("RUN_MODE", mode)
    _assignment("NODES", _positive_int(launch.get("num_nodes", 1), "launch.num_nodes"))
    _assignment(
        "GPUS_PER_NODE",
        _positive_int(launch.get("num_gpus", 1), "launch.num_gpus"),
    )
    _assignment(
        "OMP_NUM_THREADS",
        _positive_int(launch.get("omp_num_threads", 1), "launch.omp_num_threads"),
    )


def _print_slurm(cfg: dict) -> None:
    slurm = _slurm(cfg)
    _assignment("PARTITION", slurm.get("partition", "hx1hdexclu12"))
    _assignment("TIME", slurm.get("time", "24:00:00"))
    _assignment(
        "CPUS_PER_TASK",
        _positive_int(slurm.get("cpus_per_task", 16), "slurm.cpus_per_task"),
    )
    _assignment("NODELIST", slurm.get("nodelist", ""))


def _print_env(cfg: dict) -> None:
    nccl = cfg.get("nccl", {}) or {}
    if nccl.get("socket_ifname"):
        _assignment("export NCCL_SOCKET_IFNAME", nccl["socket_ifname"])
    if nccl.get("ib_hca"):
        _assignment("export NCCL_IB_HCA", nccl["ib_hca"])
    if nccl.get("proto"):
        _assignment("export NCCL_PROTO", nccl["proto"])
    print("export HSA_FORCE_FINE_GRAIN_PCIE=1")


def _print_finetune_config(cfg: dict) -> None:
    finetune = {key: value for key, value in cfg.items() if key not in META_KEYS}
    yaml.safe_dump(
        finetune,
        sys.stdout,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: _parse_config.py <config.yaml> <name|launch|slurm|env|finetune-config>"
        )
    cfg = _config(sys.argv[1])
    action = sys.argv[2]
    actions = {
        "name": lambda: print(cfg.get("name", "equiformer_v3_finetune")),
        "launch": lambda: _print_launch(cfg),
        "slurm": lambda: _print_slurm(cfg),
        "env": lambda: _print_env(cfg),
        "finetune-config": lambda: _print_finetune_config(cfg),
    }
    try:
        actions[action]()
    except KeyError as error:
        raise SystemExit(f"unknown action: {action}") from error


if __name__ == "__main__":
    main()
