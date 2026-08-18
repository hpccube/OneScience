"""Download and verify the official NequIP fcu.xyz tutorial dataset."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


URL = "https://archive.materialscloud.org/records/ycbvx-knj69/files/fcu.xyz?download=1"
SHA256 = "57f00395d6945a3018a873d229fd7fbb7352a44a66f00f3c6e8a36247e0851e5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_output() -> Path:
    datasets_dir = os.environ.get("ONESCIENCE_DATASETS_DIR")
    if not datasets_dir:
        raise RuntimeError("ONESCIENCE_DATASETS_DIR is not set; load matchem_env.sh first")
    return Path(datasets_dir) / "matchem" / "NequIP" / "fcu.xyz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Destination for fcu.xyz")
    args = parser.parse_args()
    output = (args.output or default_output()).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.is_file() and sha256(output) == SHA256:
        print(f"Using verified dataset: {output}")
        return

    request = urllib.request.Request(URL, headers={"User-Agent": "OneScience-NequIP/0.19"})
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="fcu_", suffix=".xyz.part", dir=output.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(request) as response:
                while block := response.read(1024 * 1024):
                    temporary.write(block)
        actual = sha256(temporary_path)
        if actual != SHA256:
            raise RuntimeError(f"fcu.xyz SHA256 mismatch: expected {SHA256}, got {actual}")
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    print(f"Downloaded verified dataset: {output}")


if __name__ == "__main__":
    main()
