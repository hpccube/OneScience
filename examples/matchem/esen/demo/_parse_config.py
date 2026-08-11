#!/usr/bin/env python3
"""Extract eSEN demo metadata and the native finetune configuration from YAML."""

from __future__ import annotations

import sys

import yaml


META_KEYS = {"name", "description", "launch", "slurm", "nccl"}


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
    print(f"RUN_MODE={mode}")
    print(f"NODES={launch.get('num_nodes', 1)}")
    print(f"GPUS_PER_NODE={launch.get('num_gpus', 1)}")
    print(f"OMP_NUM_THREADS={launch.get('omp_num_threads', 1)}")


def _print_slurm(cfg: dict) -> None:
    slurm = _slurm(cfg)
    print(f"PARTITION={slurm.get('partition', 'hx1hdexclu12')}")
    print(f"TIME={slurm.get('time', '24:00:00')}")
    print(f"CPUS_PER_TASK={slurm.get('cpus_per_task', 16)}")


def _print_env(cfg: dict) -> None:
    nccl = cfg.get("nccl", {}) or {}
    if nccl.get("socket_ifname"):
        print(f"export NCCL_SOCKET_IFNAME={nccl['socket_ifname']}")
    if nccl.get("ib_hca"):
        print(f"export NCCL_IB_HCA={nccl['ib_hca']}")
    if nccl.get("proto"):
        print(f"export NCCL_PROTO={nccl['proto']}")
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
        "name": lambda: print(cfg.get("name", "esen_finetune")),
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
