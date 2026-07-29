"""Run OpenStructure evaluation for Boltz predictions."""

import runpy


if __name__ == "__main__":
    runpy.run_module("onescience.metrics.boltz.run_evals", run_name="__main__")
