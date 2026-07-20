import argparse
import os
from pathlib import Path

import yaml

from onescience.utils.mattersim import FineTuneConfig, MatterSimTrainer


def _load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    # Expand environment variables in string values (e.g. ${ONESCIENCE_MODELS_DIR}).
    for key, value in list(config.items()):
        if isinstance(value, str):
            config[key] = os.path.expandvars(value)
    return config


def _build_parser(base_config: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune MatterSim with OneScience")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument(
        "--train-data-path", default=base_config.get("train_data_path")
    )
    parser.add_argument(
        "--valid-data-path", default=base_config.get("valid_data_path")
    )
    parser.add_argument("--checkpoint", default=base_config.get("checkpoint"))
    parser.add_argument("--save-path", default=base_config.get("save_path", "./results/mattersim"))
    parser.add_argument("--run-name", default=base_config.get("run_name", "onescience-mattersim"))
    parser.add_argument("--epochs", type=int, default=base_config.get("epochs", 1000))
    parser.add_argument("--batch-size", type=int, default=base_config.get("batch_size", 16))
    parser.add_argument("--lr", type=float, default=base_config.get("lr", 2e-4))
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default=base_config.get("device", "cuda")
    )
    parser.add_argument("--seed", type=int, default=base_config.get("seed", 42))
    parser.add_argument(
        "--include-stresses",
        action="store_true",
        default=base_config.get("include_stresses", False),
    )
    parser.add_argument(
        "--no-include-forces",
        action="store_false",
        dest="include_forces",
        default=base_config.get("include_forces", True),
    )
    parser.add_argument(
        "--re-normalize",
        action="store_true",
        default=base_config.get("re_normalize", False),
    )
    parser.add_argument(
        "--no-save-checkpoint",
        action="store_false",
        dest="save_checkpoint",
        default=base_config.get("save_checkpoint", True),
    )
    return parser


def main() -> None:
    # Two-phase parsing: first get --config, then use YAML defaults for the rest.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    pre_args, remaining = pre_parser.parse_known_args()

    base_config = _load_yaml_config(pre_args.config) if pre_args.config else {}
    parser = _build_parser(base_config)
    args = parser.parse_args(remaining)

    # Drop None values and the config key itself.
    kwargs = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    MatterSimTrainer(FineTuneConfig(**kwargs)).fit()


if __name__ == "__main__":
    main()
