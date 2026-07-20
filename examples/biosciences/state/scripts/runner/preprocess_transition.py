"""Preprocess a raw AnnData file for State Transition training."""

from onescience.models.state.cli._tx._preprocess_train import (
    add_arguments_preprocess_train,
    run_tx_preprocess_train,
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_preprocess_train(parser)
    args = parser.parse_args()
    run_tx_preprocess_train(args.adata, args.output, args.num_hvgs)


if __name__ == "__main__":
    main()
