"""Generate State Embeddings for a new AnnData file."""

import _bootstrap  # noqa: F401

from _cli._emb._transform import add_arguments_transform, run_emb_transform


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_transform(parser)
    run_emb_transform(parser.parse_args())


if __name__ == "__main__":
    main()
