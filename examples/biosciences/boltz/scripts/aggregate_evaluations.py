"""Aggregate Boltz benchmark evaluation files."""

import runpy


if __name__ == "__main__":
    runpy.run_module(
        "onescience.metrics.boltz.aggregate_evals",
        run_name="__main__",
    )
