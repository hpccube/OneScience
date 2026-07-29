"""Render a STATE dataset TOML template with a local dataset directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("dataset_dir")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    content = args.template.read_text(encoding="utf-8")
    content = content.replace("__STATE_DATASET_DIR__", args.dataset_dir)
    content = content.replace("__STATE_DATA_ROOT__", args.dataset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Rendered {args.output}")


if __name__ == "__main__":
    main()
