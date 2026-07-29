"""Train a State Embedding model using a YAML configuration."""

import _bootstrap  # noqa: F401

from _cli._emb._fit import add_arguments_fit, run_emb_fit


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_fit(parser)
    args = parser.parse_args()
    from omegaconf import OmegaConf

    default_config = Path(__file__).resolve().parents[2] / "configs" / "embedding" / "state-defaults.yaml"
    config = OmegaConf.load(args.conf) if args.conf else OmegaConf.load(default_config)
    run_emb_fit(config, args)


if __name__ == "__main__":
    main()
