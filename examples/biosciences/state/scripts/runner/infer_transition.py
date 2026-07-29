"""Run State Transition inference on a new AnnData file."""

import _bootstrap  # noqa: F401

from _cli._tx._infer import add_arguments_infer, run_tx_infer


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_infer(parser)
    run_tx_infer(parser.parse_args())


if __name__ == "__main__":
    main()
