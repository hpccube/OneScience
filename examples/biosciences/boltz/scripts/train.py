"""Run Boltz training from a Hydra YAML configuration."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Training YAML configuration.")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra dot-list overrides applied after the configuration.",
    )
    args = parser.parse_args()

    # Keep the heavy Lightning/Hydra graph out of --help and argument errors.
    from _boltz_training import train

    train(args.config, args.overrides)


if __name__ == "__main__":
    main()
