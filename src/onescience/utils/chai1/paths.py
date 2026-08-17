# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import dataclasses
import logging
import os
import random
from pathlib import Path

import requests
from filelock import FileLock

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "onescience" / "chai1"


def get_model_dir() -> Path:
    """Return the directory containing Chai-1 inference assets."""
    if value := os.environ.get("CHAI1_MODEL_DIR"):
        return Path(value).expanduser()
    if value := os.environ.get("CHAI_DOWNLOADS_DIR"):
        return Path(value).expanduser()
    if value := os.environ.get("ONESCIENCE_MODELS_DIR"):
        return Path(value).expanduser() / "chai-lab"
    return DEFAULT_MODEL_DIR


def set_model_dir(model_dir: str | Path) -> None:
    """Set the asset root used by subsequent Chai-1 operations."""
    os.environ["CHAI1_MODEL_DIR"] = str(Path(model_dir).expanduser().resolve())


def download_if_not_exists(http_url: str, path: Path):
    if path.exists():
        return

    with FileLock(path.with_suffix(".download_lock")):
        if path.exists():
            return  # if-lock-if sandwich to download only once
        logging.info(f"downloading {http_url}")
        tmp_path = path.with_suffix(f".download_tmp_{random.randint(10 ** 5, 10**6)}")
        with requests.get(http_url, stream=True) as response:
            response.raise_for_status()  # Check if the request was successful
            # Open a local file with the specified name
            path.parent.mkdir(exist_ok=True, parents=True)
            with tmp_path.open("wb") as file:
                # Download the file in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive new chunks
                        file.write(chunk)
    tmp_path.rename(path)
    assert path.exists()


@dataclasses.dataclass
class Downloadable:
    url: str
    relative_path: Path

    def get_path(self) -> Path:
        # downloads artifact if necessary
        path = get_model_dir() / self.relative_path
        download_if_not_exists(self.url, path=path)
        return path


cached_conformers = Downloadable(
    url="https://chaiassets.com/chai1-inference-depencencies/conformers_v1.apkl",
    relative_path=Path("conformers_v1.apkl"),
)

COMPONENT_URL = (
    "https://chaiassets.com/chai1-inference-depencencies/models_v2/{comp_key}"
)


def chai1_component(comp_key: str) -> Path:
    """
    Downloads exported model, stores in locally in the repo/downloads
    comp_key: e.g. 'trunk.pt'
    """
    assert comp_key.endswith(".pt")
    url = COMPONENT_URL.format(comp_key=comp_key)
    result = get_model_dir().joinpath("models_v2", comp_key)
    download_if_not_exists(url, result)

    return result
