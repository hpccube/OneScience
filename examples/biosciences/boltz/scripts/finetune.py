"""Fine-tune Boltz from a Lightning checkpoint."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Training YAML configuration.")
    parser.add_argument("checkpoint", help="Pretrained Lightning checkpoint.")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra dot-list overrides applied after the checkpoint settings.",
    )
    args = parser.parse_args()

    # Keep the heavy Lightning/Hydra graph out of --help and argument errors.
    from _boltz_training import train

    train(
        args.config,
        [f"pretrained={args.checkpoint}", "resume=null", *args.overrides],
    )


if __name__ == "__main__":
    main()
