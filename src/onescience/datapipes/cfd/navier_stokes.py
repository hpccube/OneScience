import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader, DistributedSampler

from onescience.datapipes.core import BaseDataset
from onescience.distributed.manager import DistributedManager


class NavierStokesDataset(BaseDataset):
    """Transolver benchmark dataset for 2D Navier-Stokes forecasting.

    The source file must contain ``u`` with shape ``[N, H, W, T]``.
    Each sample contains flattened grid coordinates, an input time window,
    and a target time window.
    """

    DOMAIN = "cfd"
    TASK = "autoregressive_forecasting"
    DATA_FORMATS = ["mat"]

    _RAW_CACHE: Dict[Path, np.ndarray] = {}

    def __init__(
        self,
        config: Union[Dict[str, Any]],
        mode: str = "train",
        normalizer_state: Optional[Dict[str, float]] = None,
    ):
        self.mode = mode
        self.normalizer_state = normalizer_state
        self.dist = DistributedManager()
        super().__init__(config)

        self._init_paths()
        self._init_data()

        if self.dist.rank != 0:
            self.logger.setLevel(logging.WARNING)

    def _init_paths(self):
        super()._init_paths()
        self.data_file = self.data_path / self.config.source.file_name
        if not self.data_file.exists():
            raise FileNotFoundError(f"Navier-Stokes data file not found: {self.data_file}")

    def _load_raw(self):
        data_file = self.data_file.resolve()
        if data_file not in self._RAW_CACHE:
            if self.dist.rank == 0:
                self.logger.info(f"Loading Navier-Stokes data from {data_file}...")
            loaded = scio.loadmat(data_file)
            if "u" not in loaded:
                raise KeyError(f"Expected variable 'u' in Navier-Stokes file: {data_file}")
            self._RAW_CACHE[data_file] = loaded["u"].astype("float32")
        return self._RAW_CACHE[data_file]

    def _init_data(self):
        raw = self._load_raw()
        data_cfg = self.config.data

        self.r1 = int(data_cfg.downsamplex)
        self.r2 = int(data_cfg.downsampley)
        self.t_in = int(data_cfg.t_in)
        self.t_out = int(data_cfg.t_out)
        self.out_dim = int(data_cfg.out_dim)
        self.ntrain = int(data_cfg.ntrain)
        self.ntest = int(data_cfg.ntest)
        self.normalize = bool(data_cfg.normalize)

        if self.mode not in ("train", "test", "val"):
            raise ValueError(f"Unknown mode: {self.mode}")
        if min(self.r1, self.r2, self.t_in, self.t_out, self.ntrain, self.ntest) < 1:
            raise ValueError("Downsampling, time windows, and split sizes must be positive")
        if raw.ndim != 4:
            raise ValueError(f"Expected u with four dimensions [N,H,W,T], got {raw.shape}")
        if raw.shape[0] < self.ntrain + self.ntest:
            raise ValueError(
                f"Requested ntrain+ntest={self.ntrain + self.ntest}, "
                f"but dataset has {raw.shape[0]} samples."
            )
        if raw.shape[-1] < self.t_in + self.t_out:
            raise ValueError(
                f"Requested t_in+t_out={self.t_in + self.t_out}, "
                f"but dataset has {raw.shape[-1]} time steps."
            )
        if self.out_dim != 1:
            raise ValueError(
                "The standard Navier-Stokes file contains one scalar field; set out_dim=1."
            )

        self.s1 = int(((raw.shape[1] - 1) / self.r1) + 1)
        self.s2 = int(((raw.shape[2] - 1) / self.r2) + 1)
        self.spatial_shape = (self.s1, self.s2)

        train_raw = raw[: self.ntrain, :: self.r1, :: self.r2, :][
            :, : self.s1, : self.s2, :
        ]
        test_raw = raw[-self.ntest :, :: self.r1, :: self.r2, :][
            :, : self.s1, : self.s2, :
        ]
        split_raw = train_raw if self.mode == "train" else test_raw
        sample_count = split_raw.shape[0]
        self.x = torch.from_numpy(
            split_raw[..., : self.t_in].reshape(
                sample_count, -1, self.t_in * self.out_dim
            )
        )
        self.y = torch.from_numpy(
            split_raw[..., self.t_in : self.t_in + self.t_out].reshape(
                sample_count, -1, self.t_out * self.out_dim
            )
        )

        self.normalizer = None
        if self.normalize:
            if self.normalizer_state is None:
                train_x = torch.from_numpy(
                    train_raw[..., : self.t_in].reshape(
                        self.ntrain, -1, self.t_in * self.out_dim
                    )
                )
                train_y = torch.from_numpy(
                    train_raw[..., self.t_in : self.t_in + self.t_out].reshape(
                        self.ntrain, -1, self.t_out * self.out_dim
                    )
                )
                solution = torch.cat((train_x.reshape(-1), train_y.reshape(-1)))
                mean = solution.mean()
                std = solution.std().clamp_min(1.0e-6)
                self.normalizer = {"mean": mean, "std": std}
            else:
                self.normalizer = {
                    "mean": torch.as_tensor(
                        self.normalizer_state["mean"], dtype=torch.float32
                    ),
                    "std": torch.as_tensor(
                        self.normalizer_state["std"], dtype=torch.float32
                    ).clamp_min(1.0e-6),
                }
            self.x = (self.x - self.normalizer["mean"]) / self.normalizer["std"]
            self.y = (self.y - self.normalizer["mean"]) / self.normalizer["std"]

        grid_x = np.linspace(0, 1, self.s2, dtype=np.float32)
        grid_y = np.linspace(0, 1, self.s1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        self.pos = torch.from_numpy(
            np.c_[grid_x.ravel(), grid_y.ravel()].astype("float32")
        )

        if self.dist.rank == 0:
            self.logger.info(
                f"[{self.mode}] Loaded {len(self.x)} samples "
                f"with spatial_shape={self.spatial_shape}."
            )

    def get_normalizer_state(self):
        if self.normalizer is None:
            return None
        return {
            "mean": float(self.normalizer["mean"].cpu()),
            "std": float(self.normalizer["std"].cpu()),
        }

    def decode_solution(self, tensor):
        if self.normalizer is None:
            return tensor
        mean = torch.as_tensor(
            self.normalizer["mean"], dtype=tensor.dtype, device=tensor.device
        )
        std = torch.as_tensor(
            self.normalizer["std"], dtype=tensor.dtype, device=tensor.device
        )
        return tensor * std + mean

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            "pos": self.pos,
            "x": self.x[idx],
            "y": self.y[idx],
        }


class NavierStokesDatapipe:
    def __init__(
        self,
        config: Dict[str, Any],
        distributed: bool = False,
        normalizer_state: Optional[Dict[str, float]] = None,
    ):
        self.config = config
        self.distributed = distributed

        self.train_dataset = NavierStokesDataset(
            copy.deepcopy(config),
            mode="train",
            normalizer_state=normalizer_state,
        )
        state = self.train_dataset.get_normalizer_state()
        self.test_dataset = NavierStokesDataset(
            copy.deepcopy(config), mode="test", normalizer_state=state
        )

    @property
    def spatial_shape(self):
        return self.train_dataset.spatial_shape

    def get_normalizer_state(self):
        return self.train_dataset.get_normalizer_state()

    def decode_solution(self, tensor):
        return self.train_dataset.decode_solution(tensor)

    def train_dataloader(self):
        sampler = (
            DistributedSampler(self.train_dataset, shuffle=True)
            if self.distributed
            else None
        )
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config.dataloader.batch_size),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=int(self.config.dataloader.num_workers),
            pin_memory=bool(self.config.dataloader.pin_memory),
        ), sampler

    def test_dataloader(self):
        sampler = (
            DistributedSampler(self.test_dataset, shuffle=False)
            if self.distributed
            else None
        )
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.config.dataloader.batch_size),
            shuffle=False,
            sampler=sampler,
            num_workers=int(self.config.dataloader.num_workers),
            pin_memory=bool(self.config.dataloader.pin_memory),
        ), sampler
