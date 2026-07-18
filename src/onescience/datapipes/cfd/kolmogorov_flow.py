import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from onescience.datapipes.core import BaseDataset
from onescience.distributed.manager import DistributedManager


class KolmogorovFlow2DDataset(BaseDataset):
    """Memory-mapped 2D Kolmogorov flow dataset.

    Source arrays are expected to be shaped ``(num_trajectories, time, H, W)``.
    Returned samples follow the CFD neural-operator convention:

    - ``pos``: ``(N, 2)``
    - ``x``: ``(N, t_in*out_dim)``
    - ``y``: ``(N, t_out*out_dim)``
    """

    DOMAIN = "cfd"
    TASK = "autoregressive_forecasting"
    DATA_FORMATS = ["npy"]

    def __init__(
        self,
        config: Union[Dict[str, Any]],
        mode: str = "train",
        normalizer_state: Optional[Dict[str, float]] = None,
    ):
        self.mode = mode
        self.normalizer_state = normalizer_state
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        self.dist = DistributedManager()
        self._raw = None
        super().__init__(config)
        self._init_paths()
        self._init_data()

    def _init_paths(self):
        super()._init_paths()
        self.data_file = self.data_path / self.config.source.file_name
        if not self.data_file.exists():
            raise FileNotFoundError(f"Kolmogorov flow data file not found: {self.data_file}")

    @property
    def raw(self):
        if self._raw is None:
            self._raw = np.load(self.data_file, mmap_mode="r")
        return self._raw

    def _init_data(self):
        raw = self.raw
        data_cfg = self.config.data

        if raw.ndim != 4:
            raise ValueError(f"Expected data shape (S, T, H, W), got {raw.shape}.")

        self.train_num = int(data_cfg.train_num)
        self.test_num = int(data_cfg.test_num)
        self.resolution = int(data_cfg.resolution)
        self.interval = int(data_cfg.interval)
        self.t_in = int(data_cfg.t_in)
        self.t_out = int(data_cfg.t_out)
        self.out_dim = int(data_cfg.out_dim)
        self.normalize = bool(data_cfg.normalize)
        self.val_mode = bool(data_cfg.get("val_mode", False))

        if self.out_dim != 1:
            raise ValueError("KolmogorovFlow2D contains one vorticity channel; set out_dim=1.")
        if self.train_num + self.test_num > raw.shape[0]:
            raise ValueError(
                f"Requested train_num+test_num={self.train_num + self.test_num}, "
                f"but dataset has {raw.shape[0]} trajectories."
            )
        if raw.shape[2] != raw.shape[3]:
            raise ValueError("KolmogorovFlow2D expects a square grid.")
        if raw.shape[2] % self.resolution != 0:
            raise ValueError(f"resolution={self.resolution} must divide raw grid size={raw.shape[2]}.")

        self.skip = raw.shape[2] // self.resolution
        self.available_steps = raw.shape[1] // self.interval
        self.seq_len = self.available_steps - (self.t_in + self.t_out) + 1
        if self.seq_len <= 0:
            raise ValueError("t_in+t_out is longer than the available downsampled time sequence.")

        if self.val_mode:
            all_ids = np.arange(self.train_num + self.test_num)
            rng = np.random.default_rng(int(data_cfg.get("split_seed", 1234)))
            rng.shuffle(all_ids)
            self.trajectory_ids = all_ids[: self.train_num] if self.mode == "train" else all_ids[self.train_num :]
        elif self.mode == "train":
            self.trajectory_ids = np.arange(self.train_num)
        else:
            self.trajectory_ids = np.arange(raw.shape[0] - self.test_num, raw.shape[0])

        self.normalizer = self._build_normalizer()
        grid = np.linspace(0.0, 2.0 * np.pi, self.resolution, endpoint=False, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(grid, grid, indexing="ij")
        self.pos = torch.from_numpy(np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1))
        self.spatial_shape = (self.resolution, self.resolution)

        if self.dist.rank == 0:
            self.logger.info(
                f"[{self.mode}] KolmogorovFlow2D trajectories={len(self.trajectory_ids)}, "
                f"seq_len={self.seq_len}, spatial_shape={self.spatial_shape}."
            )

    def _stats_file(self) -> Optional[Path]:
        stats_file = self.config.data.get("stats_file", None)
        if stats_file in (None, ""):
            return None
        path = Path(stats_file)
        return path if path.is_absolute() else Path.cwd() / path

    def _build_normalizer(self):
        if not self.normalize:
            return None
        if self.normalizer_state is not None:
            return {
                "mean": torch.as_tensor(self.normalizer_state["mean"], dtype=torch.float32),
                "std": torch.as_tensor(self.normalizer_state["std"], dtype=torch.float32).clamp_min(1e-6),
            }

        stats_file = self._stats_file()
        if stats_file is not None and stats_file.exists():
            stats = np.load(stats_file)
            return {
                "mean": torch.tensor(float(stats["mean"]), dtype=torch.float32),
                "std": torch.tensor(float(stats["std"]), dtype=torch.float32).clamp_min(1e-6),
            }

        max_trajectories = int(self.config.data.get("stats_max_trajectories", min(8, self.train_num)))
        traj_ids = np.arange(min(max_trajectories, self.train_num))
        count = 0
        total = 0.0
        total_sq = 0.0
        for traj_id in traj_ids:
            arr = np.asarray(
                self.raw[traj_id, :: self.interval, :: self.skip, :: self.skip],
                dtype=np.float64,
            )
            total += arr.sum()
            total_sq += np.square(arr).sum()
            count += arr.size
        mean = total / max(count, 1)
        var = max(total_sq / max(count, 1) - mean * mean, 1e-12)
        std = float(np.sqrt(var))

        if stats_file is not None and self.dist.rank == 0:
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(stats_file, mean=np.float32(mean), std=np.float32(std))

        return {
            "mean": torch.tensor(float(mean), dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32).clamp_min(1e-6),
        }

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
        mean = torch.as_tensor(self.normalizer["mean"], dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(self.normalizer["std"], dtype=tensor.dtype, device=tensor.device)
        return tensor * std + mean

    def __len__(self):
        return len(self.trajectory_ids) * self.seq_len

    def __getitem__(self, idx):
        traj_offset = idx // self.seq_len
        start = idx % self.seq_len
        traj_id = int(self.trajectory_ids[traj_offset])
        start_raw = start * self.interval
        end_raw = (start + self.t_in + self.t_out) * self.interval
        fields = np.asarray(
            self.raw[traj_id, start_raw:end_raw:self.interval, :: self.skip, :: self.skip],
            dtype=np.float32,
        )
        if self.normalizer is not None:
            mean = float(self.normalizer["mean"])
            std = float(self.normalizer["std"])
            fields = (fields - mean) / std

        x = fields[: self.t_in]
        y = fields[self.t_in : self.t_in + self.t_out]
        x = torch.from_numpy(np.moveaxis(x, 0, -1).reshape(-1, self.t_in * self.out_dim).copy())
        y = torch.from_numpy(np.moveaxis(y, 0, -1).reshape(-1, self.t_out * self.out_dim).copy())
        return {
            "pos": self.pos,
            "x": x,
            "y": y,
        }


class KolmogorovFlow2DDatapipe:
    def __init__(
        self,
        config: Dict[str, Any],
        distributed: bool = False,
        normalizer_state: Optional[Dict[str, float]] = None,
    ):
        self.config = config
        self.distributed = distributed
        self.train_dataset = KolmogorovFlow2DDataset(
            copy.deepcopy(config),
            mode="train",
            normalizer_state=normalizer_state,
        )
        state = self.train_dataset.get_normalizer_state()
        self.test_dataset = KolmogorovFlow2DDataset(copy.deepcopy(config), mode="test", normalizer_state=state)

    @property
    def spatial_shape(self):
        return self.train_dataset.spatial_shape

    def get_normalizer_state(self):
        return self.train_dataset.get_normalizer_state()

    def decode_solution(self, tensor):
        return self.train_dataset.decode_solution(tensor)

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.distributed else None
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config.dataloader.batch_size),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=int(self.config.dataloader.num_workers),
            pin_memory=bool(self.config.dataloader.pin_memory),
            persistent_workers=bool(self.config.dataloader.num_workers > 0),
        ), sampler

    def test_dataloader(self):
        sampler = DistributedSampler(self.test_dataset, shuffle=False) if self.distributed else None
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.config.dataloader.batch_size),
            shuffle=False,
            sampler=sampler,
            num_workers=int(self.config.dataloader.num_workers),
            pin_memory=bool(self.config.dataloader.pin_memory),
            persistent_workers=bool(self.config.dataloader.num_workers > 0),
        ), sampler
