"""Logging helpers shared by the OneScience scGPT integration."""

import logging
import sys


logger = logging.getLogger("onescience.scgpt")
if not logger.handlers:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


__all__ = ["logger"]
