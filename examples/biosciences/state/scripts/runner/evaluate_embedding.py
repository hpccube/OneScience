"""Evaluate State Embeddings on a perturbation AnnData file."""

from onescience.models.state.cli._emb._eval import add_arguments_eval, run_emb_eval


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_eval(parser)
    run_emb_eval(parser.parse_args())


if __name__ == "__main__":
    main()
