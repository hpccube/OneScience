#!/usr/bin/env python
# coding: utf-8
"""
AIFS v1.1 — Pre-training Script
=================================
Training logic strictly follows config_pretraining.yaml and anemoi-training 0.4.0.

Usage:
    python train.py
    python train.py --config conf/config.yaml
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytz
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import onescience.datapipes.climate.era5 as onescience_era5
import earthkit.regrid as ekr

from onescience.models.aifs import AIFS

# ============================================================================
# Config loader
# ============================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# ============================================================================
# Variable list builder
# ============================================================================

def build_era5_variable_list(cfg: dict) -> List[str]:
    av = cfg["aifs_variables"]
    em = cfg["era5_mapping"]
    vars_: List[str] = []
    for name in av["surface"]:
        vars_.append(em["surface"][name])
    for name in av["soil"]:
        vars_.append(em["soil"][name])
    for v in av["pressure_level"]:
        tpl = em["pressure_level"][v]
        for lvl in av["pressure_levels"]:
            vars_.append(tpl.format(level=lvl))
    for name in av["diagnostic"]:
        vars_.append(em["diagnostic"][name])
    return vars_


def _auto_correct_variable_names(required: List[str], available: set, cfg: dict) -> List[str]:
    fuzzy = cfg["era5_mapping"].get("fuzzy_fixes", {})
    corrected = list(required)
    for i, v in enumerate(corrected):
        if v not in available and v in fuzzy:
            alt = fuzzy[v]
            if alt in available:
                print(f"[INFO] Auto-corrected: '{v}' → '{alt}'")
                corrected[i] = alt
    return corrected


def _era5_to_aifs_name(era5_name: str, cfg: dict) -> Optional[str]:
    em = cfg["era5_mapping"]
    for aifs, ename in em["surface"].items():
        if ename == era5_name: return aifs
    for aifs, ename in em["soil"].items():
        if ename == era5_name: return aifs
    for v in cfg["aifs_variables"]["pressure_level"]:
        tpl = em["pressure_level"][v]
        for lvl in cfg["aifs_variables"]["pressure_levels"]:
            if tpl.format(level=lvl) == era5_name: return f"{v}_{lvl}"
    for aifs, ename in em["diagnostic"].items():
        if ename == era5_name: return aifs
    return None

# ============================================================================
# N320 interpolation
# ============================================================================

def _interp_n320(arr: np.ndarray) -> np.ndarray:
    return ekr.interpolate(arr, {"grid": (0.25, 0.25)}, {"grid": "N320"})

# ============================================================================
# ERA5 → AIFS field conversion
# ============================================================================

def build_aifs_fields_from_frames(
    frame_tm6: np.ndarray, frame_t0: np.ndarray, frame_tp6: np.ndarray,
    era5_var_list: List[str], cfg: dict,
    diag_map: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    av = cfg["aifs_variables"]
    em = cfg["era5_mapping"]
    name_to_ch = {n: i for i, n in enumerate(era5_var_list)}
    fields: Dict[str, np.ndarray] = {}

    for aifs_name in av["surface"]:
        ch = name_to_ch[em["surface"][aifs_name]]
        fields[aifs_name] = np.stack([_interp_n320(frame_tm6[ch]),
                                       _interp_n320(frame_t0[ch]),
                                       _interp_n320(frame_tp6[ch])])
    for aifs_name in av["soil"]:
        ch = name_to_ch[em["soil"][aifs_name]]
        fields[aifs_name] = np.stack([_interp_n320(frame_tm6[ch]),
                                       _interp_n320(frame_t0[ch]),
                                       _interp_n320(frame_tp6[ch])])
    for aifs_var in av["pressure_level"]:
        tpl = em["pressure_level"][aifs_var]
        for level in av["pressure_levels"]:
            ch = name_to_ch[tpl.format(level=level)]
            fields[f"{aifs_var}_{level}"] = np.stack([_interp_n320(frame_tm6[ch]),
                                                       _interp_n320(frame_t0[ch]),
                                                       _interp_n320(frame_tp6[ch])])
    _diag_map = diag_map if diag_map is not None else em["diagnostic"]
    for aifs_name in av["diagnostic"]:
        era5_name = _diag_map.get(aifs_name)
        if era5_name and era5_name in name_to_ch:
            ch = name_to_ch[era5_name]
            fields[aifs_name] = np.stack([_interp_n320(frame_tm6[ch]),
                                           _interp_n320(frame_t0[ch]),
                                           _interp_n320(frame_tp6[ch])])
    return fields

# ============================================================================
# Forcing feature computation
# ============================================================================

def compute_forcing_features(
    latitudes: np.ndarray, longitudes: np.ndarray,
    timestamps: List[datetime.datetime],
    fields: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)
    multi_step = len(timestamps)
    forcing: Dict[str, np.ndarray] = {}

    forcing["cos_latitude"] = np.tile(np.cos(lat_rad)[np.newaxis, :], (multi_step, 1))
    forcing["sin_latitude"] = np.tile(np.sin(lat_rad)[np.newaxis, :], (multi_step, 1))
    forcing["cos_longitude"] = np.tile(np.cos(lon_rad)[np.newaxis, :], (multi_step, 1))
    forcing["sin_longitude"] = np.tile(np.sin(lon_rad)[np.newaxis, :], (multi_step, 1))

    cos_jd = np.zeros((multi_step, len(latitudes)), dtype=np.float32)
    sin_jd = np.zeros_like(cos_jd)
    cos_lt = np.zeros_like(cos_jd)
    sin_lt = np.zeros_like(cos_jd)
    insol = np.zeros_like(cos_jd)

    for t_idx, ts in enumerate(timestamps):
        doy = ts.timetuple().tm_yday
        jd_angle = 2.0 * np.pi * doy / 365.25
        cos_jd[t_idx] = np.cos(jd_angle)
        sin_jd[t_idx] = np.sin(jd_angle)
        hours = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
        lt_angle = 2.0 * np.pi * hours / 24.0 + lon_rad
        cos_lt[t_idx] = np.cos(lt_angle)
        sin_lt[t_idx] = np.sin(lt_angle)
        insol[t_idx] = _compute_insolation(ts, lat_rad, lon_rad)

    forcing["cos_julian_day"] = cos_jd
    forcing["sin_julian_day"] = sin_jd
    forcing["cos_local_time"] = cos_lt
    forcing["sin_local_time"] = sin_lt
    forcing["insolation"] = insol

    for var_name in ["lsm", "z", "slor", "sdor"]:
        if var_name in fields:
            forcing[var_name] = np.tile(fields[var_name][:1], (multi_step, 1))
    return forcing


def _compute_insolation(ts: datetime.datetime, lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    ref = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=pytz.utc)
    mt = ts.timestamp()
    days = (mt - ref.timestamp()) / (24.0 * 3600.0)
    jc = days / 36525.0
    theta = (67310.54841 + jc * (876600.0 * 3600.0 + 8640184.812866
             + jc * (0.093104 - jc * 6.2e-5)))
    gmst = np.fmod((theta / 240.0) * np.pi / 180.0, 2.0 * np.pi)
    ma = np.deg2rad(357.52910 + 35999.05030 * jc - 0.0001559 * jc**2 - 0.00000048 * jc**3)
    ml = np.deg2rad(280.46645 + 36000.76983 * jc + 0.0003032 * jc**2)
    dl = np.deg2rad((1.914600 - 0.004817 * jc - 0.000014 * jc**2) * np.sin(ma)
                     + (0.019993 - 0.000101 * jc) * np.sin(2.0 * ma)
                     + 0.000290 * np.sin(3.0 * ma))
    tl = ml + dl
    eps = np.deg2rad(23.0 + 26.0 / 60.0 + 21.406 / 3600.0
                     - (46.836769 * jc - 0.0001831 * jc**2 + 0.00200340 * jc**3
                        - 0.576e-6 * jc**4 - 4.34e-8 * jc**5) / 3600.0)
    x = np.cos(tl); y = np.cos(eps) * np.sin(tl); z = np.sin(eps) * np.sin(tl)
    r = np.sqrt(1.0 - z * z)
    dec = np.arctan2(z, r)
    ra = 2.0 * np.arctan2(y, x + r)
    ha = (gmst + lon_rad) - ra
    cos_z = np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.cos(ha)
    return cos_z.astype(np.float32)

# ============================================================================
# Normalisation
# ============================================================================

def compute_normalisation_params(
    cfg: dict, variable_names: List[str],
    statistics_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-variable norm_mul and norm_add from config.

    If ``statistics_path`` points to a ``.npz`` file with keys
    ``mean`` and ``stdev`` (per-variable arrays matching the full
    dataset variable order), those values are used.  Otherwise
    identity (mean=0, stdev=1) is assumed — correct for fake data
    but NOT for real ERA5.
    """
    nc = cfg["normalizer"]
    default_method = nc["default"]
    remap = nc.get("remap", {}) or {}
    methods: Dict[str, str] = {}
    for v in variable_names:
        if v in (nc.get("none") or []): methods[v] = "none"
        elif v in (nc.get("max") or []): methods[v] = "max"
        elif v in (nc.get("min-max") or []): methods[v] = "min-max"
        elif v in (nc.get("std") or []): methods[v] = "std"
        else: methods[v] = default_method

    name_to_idx = {n: i for i, n in enumerate(variable_names)}
    num_vars = len(variable_names)

    # Load pre-computed statistics if available, else identity
    _mean = np.zeros(num_vars, dtype=np.float32)
    _stdev = np.ones(num_vars, dtype=np.float32)

    if statistics_path and os.path.exists(statistics_path):
        stats = np.load(statistics_path)
        # statistics file stores arrays indexed by its own variable list
        if "variables" in stats:
            ds_vars = [str(v) for v in stats["variables"]]
        else:
            # Fallback: assume 115-variable dataset order from aifs_config
            import json
            import onescience.models.aifs as aifs_pkg
            cfg_path = os.path.join(os.path.dirname(aifs_pkg.__file__), "aifs_config.json")
            ds_vars = json.load(open(cfg_path))["dataset"]["variables"]
        ds_name_to_idx = {n: i for i, n in enumerate(ds_vars)}
        for name, i in name_to_idx.items():
            if name in ds_name_to_idx:
                j = ds_name_to_idx[name]
                _mean[i] = float(stats["mean"][j])
                _stdev[i] = float(stats["stdev"][j])
        print(f"[INFO] Loaded statistics from {statistics_path}")
    else:
        print("[INFO] No statistics file — using identity normalisation "
              "(mean=0, stdev=1).  Fine for fake data, "
              "WRONG for real ERA5.")

    for target_var, source_var in remap.items():
        if target_var in name_to_idx and source_var in name_to_idx:
            ti, si = name_to_idx[target_var], name_to_idx[source_var]
            _mean[ti], _stdev[ti] = _mean[si], _stdev[si]

    norm_mul = np.ones(num_vars, dtype=np.float32)
    norm_add = np.zeros(num_vars, dtype=np.float32)
    for name, i in name_to_idx.items():
        method = methods.get(name, default_method)
        if method == "mean-std":
            norm_mul[i] = 1.0 / max(float(_stdev[i]), 1e-9)
            norm_add[i] = -float(_mean[i]) / max(float(_stdev[i]), 1e-9)
        elif method == "std":
            norm_mul[i] = 1.0 / max(float(_stdev[i]), 1e-9)
        elif method in ("max", "min-max", "none"):
            norm_mul[i] = 1.0
    return norm_mul, norm_add


def compute_era5_statistics(
    data_dir: str, years: List[int],
    variable_names: List[str],
    output_path: str,
) -> None:
    """Compute per-variable mean and stdev from ERA5 H5 files.

    Saves ``mean`` and ``stdev`` arrays (indexed by *variable_names*
    order) to *output_path* as a ``.npz`` file.

    This is a one-time pre-computation step before training with
    real ERA5 data.  The resulting file is consumed by
    ``compute_normalisation_params(statistics_path=...)``.
    """
    import h5py

    num_vars = len(variable_names)
    sum_x = np.zeros(num_vars, dtype=np.float64)
    sum_x2 = np.zeros(num_vars, dtype=np.float64)
    count = 0

    for year in years:
        h5_path = os.path.join(data_dir, "data", f"{year}.h5")
        if not os.path.exists(h5_path):
            print(f"[WARN] Missing {h5_path}, skipping")
            continue

        with h5py.File(h5_path, "r") as f:
            ds = f["fields"]
            h5_vars = [v.decode() if isinstance(v, bytes) else v
                       for v in ds.attrs["variables"]]
            T = ds.shape[0]

            # Index mapping for each timestep
            for t in range(T):
                frame = ds[t]  # (C, H, W)
                for i, vname in enumerate(variable_names):
                    if vname in h5_vars:
                        ch = h5_vars.index(vname)
                        vals = frame[ch].astype(np.float64)
                        sum_x[i] += vals.mean()
                        sum_x2[i] += (vals ** 2).mean()
                count += 1

    mean = (sum_x / count).astype(np.float32)
    # stdev = sqrt(E[X²] - E[X]²)
    var = np.maximum(sum_x2 / count - mean.astype(np.float64) ** 2, 0)
    stdev = np.sqrt(var).astype(np.float32)

    np.savez(output_path, mean=mean, stdev=stdev)
    print(f"[INFO] Statistics saved to {output_path} "
          f"({count} frames, {num_vars} vars)")

# ============================================================================
# Loss scaling
# ============================================================================

def build_loss_scale_factors(cfg: dict, output_var_names: List[str]) -> np.ndarray:
    sc = cfg["loss_scaling"]
    default = sc["default"]; pl = sc["pl"]; sfc = sc["sfc"]
    av = cfg["aifs_variables"]
    name_to_scale: Dict[str, float] = {}
    for var_short in av["pressure_level"]:
        base = pl.get(var_short, default)
        for lvl in av["pressure_levels"]:
            name_to_scale[f"{var_short}_{lvl}"] = base
    for var_name in av["surface"] + av["soil"] + av["diagnostic"]:
        name_to_scale[var_name] = sfc.get(var_name, default)
    for k, v in sfc.items():
        if k not in name_to_scale: name_to_scale[k] = v
    return np.array([name_to_scale.get(n, default) for n in output_var_names], dtype=np.float32)


def build_pressure_level_scaler(cfg: dict, output_var_names: List[str]) -> np.ndarray:
    pls = cfg["loss_scaling"]["pressure_level_scaler"]
    minimum, slope = pls["minimum"], pls["slope"]
    av = cfg["aifs_variables"]
    factors = np.ones(len(output_var_names), dtype=np.float32)
    for i, name in enumerate(output_var_names):
        for aifs_var in av["pressure_level"]:
            if name.startswith(f"{aifs_var}_"):
                plev_hpa = int(name[len(aifs_var) + 1:])
                factors[i] = max(minimum, slope * plev_hpa)
                break
    return factors

# ============================================================================
# Dataset
# ============================================================================

class AIFSTrainingDataset:
    def __init__(self, dataset_path: str, years: List[int], cfg: dict,
                 model_input_order: List[str], model_output_order: List[str],
                 norm_mul: np.ndarray, norm_add: np.ndarray,
                 input_steps=2, output_steps=1,
                 grid_lat=None, grid_lon=None):
        dp = dataset_path.rstrip("/")
        if os.path.isdir(os.path.join(dp, "data")): era5_root = dp
        elif os.path.isdir(dp) and any(f.endswith(".h5") for f in os.listdir(dp)): era5_root = os.path.dirname(dp)
        else: era5_root = dp
        self._era5_root = era5_root

        data_dir = os.path.join(era5_root, "data")
        h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])
        era5_var_list = build_era5_variable_list(cfg)
        available_set: set = set()

        if h5_files:
            import h5py
            with h5py.File(os.path.join(data_dir, h5_files[0]), "r") as f:
                available_vars = [v.decode() if isinstance(v, bytes) else v for v in f["fields"].attrs["variables"]]
            available_set = set(available_vars)
            corrected = _auto_correct_variable_names(era5_var_list, available_set, cfg)
            used_vars = [v for v in corrected if v in available_set]
            missing = [v for v in corrected if v not in available_set]
            if missing:
                print(f"[WARN] {len(missing)}/{len(era5_var_list)} vars missing from H5: {missing[:10]}")
        else:
            used_vars = era5_var_list

        self._cfg = cfg
        self.era5_var_list = used_vars
        self.model_input_order = model_input_order
        self.model_output_order = model_output_order
        self.norm_mul = norm_mul; self.norm_add = norm_add
        self.input_steps = input_steps; self.output_steps = output_steps
        self._name_to_input_idx = {n: i for i, n in enumerate(model_input_order)}
        self._name_to_output_idx = {n: i for i, n in enumerate(model_output_order)}

        # corrected diagnostic map
        self._diag_map: Dict[str, str] = {}
        for aifs_name, orig in cfg["era5_mapping"]["diagnostic"].items():
            cn = _auto_correct_variable_names([orig], available_set, cfg)[0] if h5_files else orig
            self._diag_map[aifs_name] = cn if (h5_files and cn in available_set) else orig

        self._dataset = onescience_era5.ERA5Dataset(
            dataset_dir=era5_root, used_years=years,
            used_variables=used_vars, input_steps=input_steps,
            output_steps=output_steps, normalize=False)

        self._grid_lat = grid_lat; self._grid_lon = grid_lon
        self.n_samples = len(self._dataset)

        # audit
        self._print_audit(cfg, used_vars)

    def _print_audit(self, cfg, used_vars):
        av = cfg["aifs_variables"]; em = cfg["era5_mapping"]
        fields_ok = set()
        for aifs in av["surface"]:
            if em["surface"][aifs] in used_vars: fields_ok.add(aifs)
        for aifs in av["soil"]:
            if em["soil"][aifs] in used_vars: fields_ok.add(aifs)
        for vv in av["pressure_level"]:
            for lvl in av["pressure_levels"]:
                if em["pressure_level"][vv].format(level=lvl) in used_vars: fields_ok.add(f"{vv}_{lvl}")
        for aifs in av["diagnostic"]:
            if self._diag_map[aifs] in used_vars: fields_ok.add(aifs)
        forcing_ok = set(av["computed_forcing"] + ["lsm", "z", "slor", "sdor"])
        mi = [v for v in self.model_input_order if v not in fields_ok and v not in forcing_ok]
        mo = [v for v in self.model_output_order if v not in fields_ok]
        print(f"[INFO] Variables: {len(fields_ok)} fields + {len(forcing_ok)} forcing = all available")
        if mi: print(f"[WARN] {len(mi)} input vars zero-filled: {mi}")
        if mo: print(f"[WARN] {len(mo)} output vars no target data: {mo}")

    def __len__(self): return self.n_samples
    def set_grid(self, lat, lon):
        self._grid_lat = lat.astype(np.float32); self._grid_lon = lon.astype(np.float32)

    def _get_raw_sample(self, idx: int):
        result = self._dataset[int(idx)]
        return result[0], result[1], int(result[3]), list(result[4])

    def process_sample(self, invar, outvar, step_idx, time_index):
        if invar.ndim == 3: invar = invar[np.newaxis, ...]
        if outvar.ndim == 3: outvar = outvar[np.newaxis, ...]
        total_steps = self.input_steps + self.output_steps
        all_frames = np.concatenate([invar, outvar], axis=0)
        timestamps = [datetime.datetime.strptime(t, "%Y%m%d%H").replace(tzinfo=pytz.utc) for t in time_index]

        frame_tm6, frame_t0, frame_tp6 = all_frames[0], all_frames[1], all_frames[2] if total_steps >= 3 else all_frames[-1]
        fields = build_aifs_fields_from_frames(frame_tm6, frame_t0, frame_tp6, self.era5_var_list, self._cfg, diag_map=self._diag_map)

        num_pts = next(iter(fields.values())).shape[-1]
        if self._grid_lat is None:
            self._grid_lat = np.linspace(90, -90, 542080, dtype=np.float32)[:num_pts]
            self._grid_lon = np.linspace(0, 360, 542080, dtype=np.float32)[:num_pts]

        forcing = compute_forcing_features(self._grid_lat, self._grid_lon, timestamps, fields)

        num_input_vars = len(self.model_input_order)
        input_tensor = np.zeros((self.input_steps, num_pts, num_input_vars), dtype=np.float32)
        for var_name, var_idx in self._name_to_input_idx.items():
            for t in range(self.input_steps):
                if var_name in fields: input_tensor[t, :, var_idx] = fields[var_name][t]
                elif var_name in forcing: input_tensor[t, :, var_idx] = forcing[var_name][t]

        input_tensor = input_tensor * self.norm_mul[np.newaxis, np.newaxis, :] + self.norm_add[np.newaxis, np.newaxis, :]

        num_output_vars = len(self.model_output_order)
        target_tensor = np.zeros((1, num_pts, num_output_vars), dtype=np.float32)
        t_target = self.input_steps
        for var_name, var_idx in self._name_to_output_idx.items():
            if var_name in fields: target_tensor[0, :, var_idx] = fields[var_name][t_target]
        return input_tensor, target_tensor

# ============================================================================
# Model loading
# ============================================================================

def load_aifs_model(checkpoint_path: str, device: str, pretrained: bool = False):
    """Load the AIFS model through the onescience wrapper.

    Parameters
    ----------
    pretrained : bool
        False (default): Build model from scratch via ``AIFS.from_scratch()``.
        Uses local static config + grid files — NO checkpoint needed.
        Equivalent to FengWu/Fuxi ``model = Fengwu()`` pattern.
        True: Load serialised model with pretrained weights from .ckpt.
    """
    t0 = time.time()
    if pretrained:
        model = AIFS(checkpoint_path, device=device, pretrained=True)
    else:
        model = AIFS.from_scratch(device=device)
    mode = "pretrained" if pretrained else "from scratch"
    print(f"[INFO] Model loaded ({mode}, {time.time() - t0:.1f}s)")

    input_vars  = model.input_variables
    output_vars = model.output_variables
    grid_lat    = model.latitudes.cpu().numpy()
    grid_lon    = model.longitudes.cpu().numpy()
    node_wts    = model.node_weights   # np.ndarray or None
    return model, input_vars, output_vars, grid_lat, grid_lon, node_wts

# ============================================================================
# Loss
# ============================================================================

class WeightedMSELoss(nn.Module):
    def __init__(self, var_scale: np.ndarray, pl_scale: np.ndarray, node_weights=None):
        super().__init__()
        self.register_buffer("combined_scale", torch.from_numpy((var_scale * pl_scale).astype(np.float32)))
        if node_weights is not None:
            self.register_buffer("node_weights", torch.from_numpy(node_weights.astype(np.float32)))
        else: self.node_weights = None

    def forward(self, pred, target):
        sq_error = (pred - target) ** 2
        scaled = sq_error * self.combined_scale[None, None, :]
        if self.node_weights is not None:
            x = scaled.mean(dim=-1) * self.node_weights[None, :]
            return (x / self.node_weights.sum()).sum()
        return scaled.mean()

# ============================================================================
# LR schedule
# ============================================================================

def cosine_lr_schedule(step, warmup_steps, total_steps, peak_lr, min_lr):
    """Cosine LR with linear warmup.

    Manual implementation equivalent to ``timm.scheduler.CosineLRScheduler``
    for a single decay cycle (no restarts).  Same formula as the official
    anemoi-training configuration.
    """
    if step < warmup_steps:
        return peak_lr * (step / max(warmup_steps, 1))
    progress = min(
        (step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0,
    )
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

# ============================================================================
# Distributed helpers
# ============================================================================

def _init_distributed(hw: dict) -> dict:
    """初始化分布式训练环境。

    支持三种模式：
    - 单卡：直接运行 ``python train.py``
    - 单机多卡：``torchrun --nproc_per_node=N train.py``
    - 集群多机：``torchrun --nnodes=M --nproc_per_node=N ...``
    """
    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if rank >= 0 and world_size > 1:
        # torchrun / torch.distributed.launch 模式
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
        print(f"[INFO] Distributed: rank={rank}/{world_size}, "
              f"local_rank={local_rank}, device={device_str}")
    elif world_size > 1 and "MASTER_ADDR" in os.environ:
        # 手动设置的环境变量（slurm 等）
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "device": device_str,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def _wrap_model_for_distributed(model, device_id: int):
    """用 DDP 包装模型（多卡时），并返回可用于前向传播的模型引用。

    注意：AIFS 的图结构（HeteroData）在所有 rank 上相同，只有
    模型参数通过 DDP 的 all-reduce 同步梯度。数据并行通过
    DistributedSampler 实现。
    """
    if torch.distributed.is_initialized():
        model._model = torch.nn.parallel.DistributedDataParallel(
            model._model,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=False,
        )
        # DDP 包装后，forward 调用不变，但 backward 会自动同步梯度
        print(f"[INFO] DDP wrapped (device={device_id})")
    return model


# ============================================================================
# Training
# ============================================================================

def train(cfg: dict):
    hw = cfg["hardware"]; data = cfg["data"]; ck = cfg["checkpoint"]; tr = cfg["training"]

    # ---- 分布式初始化 ------------------------------------------------
    dist_info = _init_distributed(hw)
    device = dist_info["device"]
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    is_main = (rank == 0)

    if is_main:
        print(f"[INFO] Device: {device}  |  GPUs: {world_size}")

    train_years = data["train_years"]
    val_years = data["val_years"]

    from_scratch = tr.get("from_scratch", True)
    model, model_input_vars, model_output_vars, grid_lat, grid_lon, node_wts = \
        load_aifs_model(ck.get("pretrained", ""), device,
                        pretrained=not from_scratch)

    # DDP 包装（多卡时同步梯度）
    if world_size > 1:
        model = _wrap_model_for_distributed(model, dist_info["local_rank"])

    # ---- 自动计算归一化统计量（如需要）-------------------------------
    stats_path = cfg["normalizer"].get("statistics_path") or None
    if stats_path:
        stats_path = os.path.join(str(ROOT), stats_path)
        if not os.path.exists(stats_path) and is_main:
            # 自动计算 —— 对标 anemoi-training DataModule 的统计量阶段
            print("[INFO] Statistics file not found — auto-computing ...")
            ds_vars = None
            if hasattr(model, '_meta') and 'dataset' in model._meta:
                ds_vars = model._meta['dataset']['variables']
            elif hasattr(model, '_meta') and 'dataset' in model._meta:
                ds_vars = model._meta['dataset']['variables']
            else:
                import onescience.models.aifs as aifs_pkg
                cfg_json_path = os.path.join(os.path.dirname(aifs_pkg.__file__), "aifs_config.json")
                if os.path.exists(cfg_json_path):
                    import json
                    ds_vars = json.load(
                        open(cfg_json_path)
                    )['dataset']['variables']
            if ds_vars is None:
                print("[WARN] Cannot determine dataset variables — "
                      "falling back to identity normalisation")
                stats_path = None
            else:
                os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                compute_era5_statistics(
                    data["data_dir"], train_years + val_years,
                    ds_vars, stats_path,
                )

    norm_mul, norm_add = compute_normalisation_params(
        cfg, model_input_vars,
        statistics_path=stats_path,
    )

    var_scale = build_loss_scale_factors(cfg, model_output_vars)
    pl_scale = build_pressure_level_scaler(cfg, model_output_vars)

    criterion = WeightedMSELoss(var_scale, pl_scale, node_wts).to(device)
    val_criterion = WeightedMSELoss(
        np.ones(len(model_output_vars), dtype=np.float32),
        np.ones(len(model_output_vars), dtype=np.float32), node_wts).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0,
        betas=tr["optimizer"]["betas"],
        weight_decay=tr["optimizer"].get("weight_decay", 0.01),
    )

    train_dataset = AIFSTrainingDataset(
        data["data_dir"], train_years, cfg, model_input_vars, model_output_vars,
        norm_mul, norm_add, input_steps=2, output_steps=1,
        grid_lat=grid_lat, grid_lon=grid_lon)
    train_dataset.set_grid(grid_lat, grid_lon)
    val_dataset = AIFSTrainingDataset(
        data["data_dir"], val_years, cfg, model_input_vars, model_output_vars,
        norm_mul, norm_add, input_steps=2, output_steps=1,
        grid_lat=grid_lat, grid_lon=grid_lon)
    val_dataset.set_grid(grid_lat, grid_lon)

    os.makedirs(ck["output_dir"], exist_ok=True)
    use_amp = (str(device).startswith("cuda"))
    amp_dtype = getattr(torch, tr.get("amp_dtype", "float16"))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model.train()
    global_step, epoch, best_val = 0, 0, float("inf")
    t_start = time.time()
    pbar = tqdm(total=tr["max_steps"], desc=f"Training[rank={rank}]", unit="step",
                dynamic_ncols=True, disable=not is_main)

    while global_step < tr["max_steps"]:
        epoch += 1
        indices = np.random.permutation(len(train_dataset))
        for batch_start in range(0, len(train_dataset), tr["batch_size"]):
            batch_indices = indices[batch_start:batch_start + tr["batch_size"]]
            batch_inputs, batch_targets = [], []
            for idx in batch_indices:
                try:
                    invar, outvar, si, ti = train_dataset._get_raw_sample(idx)
                    invar_np = invar.numpy() if hasattr(invar, "numpy") else np.asarray(invar)
                    outvar_np = outvar.numpy() if hasattr(outvar, "numpy") else np.asarray(outvar)
                    inp, tgt = train_dataset.process_sample(invar_np, outvar_np, si, ti)
                    batch_inputs.append(inp); batch_targets.append(tgt)
                except Exception as e:
                    print(f"[WARN] Skip sample {idx}: {e}")
            if not batch_inputs: continue

            x = torch.from_numpy(np.stack(batch_inputs, axis=0)).unsqueeze(2).to(device)
            y = torch.from_numpy(np.stack(batch_targets, axis=0)).squeeze(1).to(device)

            with (torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp
                  else torch.no_grad()):
                pred = model(x)                 # AIFS.forward already squeezes dim

            loss = criterion(pred, y) / tr["accum_grad_batches"]
            (scaler.scale(loss) if scaler else loss).backward()

            # optimizer step
            if (batch_start // tr["batch_size"] + 1) % tr["accum_grad_batches"] == 0:
                if scaler: scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), tr["gradient_clip_val"])
                (scaler.step(optimizer) if scaler else optimizer.step())
                if scaler: scaler.update()
                optimizer.zero_grad()
                global_step += 1

                lr = cosine_lr_schedule(global_step, tr["warmup_steps"], tr["max_steps"], tr["peak_lr"], tr["min_lr"])
                for pg in optimizer.param_groups: pg["lr"] = lr

                elapsed = time.time() - t_start
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}", "stp/s": f"{global_step/max(elapsed,1):.1f}"})
                pbar.update(1)

                if is_main and global_step % tr["val_interval"] == 0:
                    vl = validate(model, val_dataset, val_criterion, device)
                    pbar.write(f"[step {global_step:06d}] VAL loss={vl:.6f}")
                    model.train()
                    if vl < best_val:
                        best_val = vl
                        _save(model, optimizer, global_step, epoch, vl,
                              ck["output_dir"], "best")
                if is_main and global_step % tr["save_interval"] == 0:
                    _save(model, optimizer, global_step, epoch, loss.item(),
                          ck["output_dir"], f"step{global_step:06d}")
                if global_step >= tr["max_steps"]: break

    if is_main:
        pbar.close()
        _save(model, optimizer, global_step, epoch, loss.item(),
              ck["output_dir"], "final")
        print(f"\n[INFO] Done: {global_step} steps in "
              f"{(time.time()-t_start)/60:.1f}min, best_val={best_val:.6f}")
    else:
        pbar.close()

@torch.no_grad()
def validate(model, dataset, criterion, device):
    model.eval()
    total, n = 0.0, 0
    use_amp = str(device).startswith("cuda")
    amp_dtype_val = torch.float16
    _first_err = True
    for idx in range(len(dataset)):
        try:
            invar, outvar, si, ti = dataset._get_raw_sample(idx)
            invar_np = invar.numpy() if hasattr(invar, "numpy") else np.asarray(invar)
            outvar_np = outvar.numpy() if hasattr(outvar, "numpy") else np.asarray(outvar)
            inp, tgt = dataset.process_sample(invar_np, outvar_np, si, ti)
            x = torch.from_numpy(inp).unsqueeze(0).unsqueeze(2).to(device)
            y = torch.from_numpy(tgt).squeeze(1).unsqueeze(0).to(device)
            with (torch.amp.autocast("cuda", dtype=amp_dtype_val) if use_amp
                  else torch.no_grad()):
                pred = model(x)                 # AIFS.forward already squeezes dim
            total += criterion(pred, y).item(); n += 1
        except Exception as e:
            if _first_err:
                print(f"[WARN] Validation error on sample {idx}: {e}")
                _first_err = False
    return total / max(n, 1)

def _save(model, optimizer, step, epoch, loss, out_dir, tag):
    """Save training checkpoint — 推理+续训合一，零外部依赖.

    单个 .ckpt 文件包含:
    - 完整 AnemoiModelEncProcDec (torch.load 直接得模型 → SimpleRunner 可用)
    - ai-models.json + supporting arrays (inference metadata)
    - optimizer.pkl (恢复训练用，推理时忽略)
    """
    import pickle, zipfile

    path = os.path.join(out_dir, f"model_bak.ckpt")

    # ---- 1.  构建 AnemoiModelInterface（含 predict_step）---------
    from anemoi.models.interface import AnemoiModelInterface
    import numpy as np

    # DDP 包装时需解包底层模型
    inner = model._model
    if hasattr(inner, "module"):  # DDP wrapper
        inner = inner.module

    # 虚拟统计量（identity normalization：mean=0, stdev=1）
    num_vars = len(model._interface_data_indices.data.input.name_to_index)
    dummy_stats = {
        "minimum": np.zeros(num_vars, dtype=np.float32),
        "maximum": np.ones(num_vars, dtype=np.float32),
        "mean": np.zeros(num_vars, dtype=np.float32),
        "stdev": np.ones(num_vars, dtype=np.float32),
    }

    interface = AnemoiModelInterface(
        config=model._interface_config,
        graph_data=model._interface_graph_data,
        statistics=dummy_stats,
        data_indices=model._interface_data_indices,
        metadata={},
        truncation_data=getattr(inner, '_truncation_data', {}),
    )
    interface.model.load_state_dict(inner.state_dict())
    interface.cpu()
    torch.save(interface, path)
    del interface

    # ---- 2.  注入 metadata (从本地静态文件) ---------------------------
    _inject_metadata(path)

    # ---- 3.  追加 optimizer 到同一子目录 ----------------------------
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/data.pkl") or name.endswith("/byteorder"):
                base_dir = name.split("/")[0]
                break
        else:
            base_dir = "checkpoint"

    opt_data = {
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "epoch": epoch, "loss": loss,
    }
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr(f"{base_dir}/optimizer.pkl", pickle.dumps(opt_data))

    print(f"[INFO] Saved: {path}")


def _inject_metadata(dst_path: str):
    """从项目静态文件生成完整 metadata 并注入 checkpoint ZIP。

    自给自足——不依赖原始 .ckpt。metadata 写入 torch.save 使用的
    同一个子目录，保持 PyTorch ZIP 结构兼容。
    """
    import json, zipfile
    import numpy as np
    import onescience.models.aifs as aifs_pkg

    _aifs_dir = os.path.dirname(aifs_pkg.__file__)
    config_path = os.path.join(_aifs_dir, "aifs_config.json")
    grid_path = aifs_pkg.aifs._find_grid_file()

    with open(config_path) as f:
        config_data = json.load(f)
    grid = np.load(grid_path)

    # ---- 1. 探测 torch.save 使用的子目录 -------------------------------
    with zipfile.ZipFile(dst_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/data.pkl") or name.endswith("/byteorder"):
                base_dir = name.split("/")[0]
                break
        else:
            base_dir = "checkpoint"
        meta_dir = f"{base_dir}/anemoi-metadata"

    # ---- 2. 构建完整 metadata -------------------------------------------
    metadata = {
        "config": config_data["model_config"],
        "data_indices": config_data["data_indices"],
        "dataset": config_data["dataset"],
        "provenance_training": config_data.get("provenance_training", {}),
        "training": config_data.get("training", {}),
        "run_id": config_data.get("run_id", ""),
        "seed": config_data.get("seed", 0),
        "uuid": config_data.get("uuid", ""),
        "timestamp": config_data.get("timestamp", ""),
        "version": "1.0",
        "supporting_arrays_paths": {
            "latitudes": {
                "path": f"{meta_dir}/latitudes.numpy",
                "shape": [542080],
                "dtype": "float64",
            },
            "longitudes": {
                "path": f"{meta_dir}/longitudes.numpy",
                "shape": [542080],
                "dtype": "float64",
            },
        },
    }

    # ---- 3. 写入 checkpoint ZIP（与模型同一子目录）---------------------
    meta_json = json.dumps(metadata).encode("utf-8")
    with zipfile.ZipFile(dst_path, "a") as dst:
        dst.writestr(f"{meta_dir}/ai-models.json", meta_json)
        dst.writestr(
            f"{meta_dir}/latitudes.numpy",
            grid["latitudes"].astype(np.float64).tobytes(),
        )
        dst.writestr(
            f"{meta_dir}/longitudes.numpy",
            grid["longitudes"].astype(np.float64).tobytes(),
        )

    print(f"[INFO] Metadata injected into {base_dir}/")

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AIFS v1.1 Training")
    parser.add_argument("--config", "-c", type=str,
                        default=str(ROOT / "conf" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    hw = cfg["hardware"]

    # 单卡模式：设置可见设备；多卡模式：torchrun 已设置 CUDA_VISIBLE_DEVICES
    if int(os.environ.get("WORLD_SIZE", 1)) <= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hw["device_ids"])
        if hw.get("device") == "dcu":
            os.environ["HIP_VISIBLE_DEVICES"] = str(hw["device_ids"])
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    seed = cfg["training"]["seed"]
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        n_gpus = max(int(os.environ.get("WORLD_SIZE", 1)), 1)
        print(f"\n{'='*60}\n  AIFS v1.1 — Pre-training\n{'='*60}")
        print(f"  Device: {hw.get('device', 'cuda')}  "
              f"GPUs: {n_gpus}  Dataset: {cfg['data']['data_dir']}")
        print(f"  LR: {cfg['training']['peak_lr']:.2e}  "
              f"Steps: {cfg['training']['max_steps']}")
        print(f"  Output: {cfg['checkpoint']['output_dir']}\n{'='*60}\n")

    try:
        train(cfg)
    except KeyboardInterrupt:
        if rank == 0: print("\n[INFO] Interrupted.")
    except Exception:
        traceback.print_exc(); sys.exit(1)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__": main()
