"""NequIP training entry point for OneScience.

This is a thin wrapper around ``onescience.utils.nequip.cli.train``. OneScience
must already be installed in the active MatChem environment.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="e3nn")

from onescience.utils.nequip.cli.train import main


if __name__ == "__main__":
    main()
