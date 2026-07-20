"""Train a State Transition model from the example configuration tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir

from onescience.models.state.cli._tx._train import run_tx_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, for example model=state training.max_steps=10",
    )
    parser.add_argument(
        "--config-dir",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "transition"),
        help="Directory containing transition/config.yaml and Hydra config groups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with initialize_config_dir(version_base=None, config_dir=args.config_dir):
        cfg = compose(config_name="config", overrides=args.overrides)
    run_tx_train(cfg)


if __name__ == "__main__":
    main()
