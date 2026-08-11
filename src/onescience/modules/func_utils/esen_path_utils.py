"""Path helpers for eSEN runtime assets."""

from __future__ import annotations

import os
from pathlib import Path

from onescience.modules.func_utils.uma_path_utils import resolve_jd_path


def resolve_esen_jd_path(jd_path: str | None = None) -> str:
    """Resolve eSEN's shared ``Jd.pt`` without changing UMA precedence."""

    candidates: list[Path] = []
    if jd_path:
        candidates.append(Path(jd_path))

    env_path = os.environ.get("ONESCIENCE_ESEN_JD_PATH")
    if env_path:
        candidates.append(Path(env_path))

    models_dir = os.environ.get("ONESCIENCE_MODELS_DIR")
    if models_dir:
        candidates.append(Path(models_dir) / "UMA" / "checkpoint" / "Jd.pt")

    for path in candidates:
        if path.is_file():
            return str(path)

    return resolve_jd_path()
