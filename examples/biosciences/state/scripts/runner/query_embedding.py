"""Query a LanceDB State Embedding index."""

from onescience.models.state.cli._emb._query import add_arguments_query, run_emb_query


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_query(parser)
    run_emb_query(parser.parse_args())


if __name__ == "__main__":
    main()
