"""Evaluate or run prediction from a trained State Transition run."""

import _bootstrap  # noqa: F401

from _cli._tx._predict import add_arguments_predict, run_tx_predict


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_predict(parser)
    run_tx_predict(parser.parse_args())


if __name__ == "__main__":
    main()
