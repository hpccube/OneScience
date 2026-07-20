"""Create a State Embedding data profile."""

from onescience.models.state.cli._emb._preprocess import add_arguments_preprocess, run_emb_preprocess


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_preprocess(parser)
    run_emb_preprocess(parser.parse_args())


if __name__ == "__main__":
    main()
