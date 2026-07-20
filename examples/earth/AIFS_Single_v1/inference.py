#!/usr/bin/env python
# coding: utf-8
"""
AIFS v1.1 — Inference Script  (AIFS-powered)
===============================================
10-day deterministic forecast from ERA5 initial conditions.

Uses ``onescience.models.aifs.AIFS`` for autoregressive forecast — variable ordering,
normalisation, and model forward are guaranteed to match training.

Usage:
    python inference.py
    python inference.py --checkpoint weights/aifs_pretrain_final.ckpt
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytz
import torch
import yaml
from tqdm import tqdm

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import earthkit.regrid as ekr
from onescience.datapipes.climate.era5 import ERA5Dataset
from onescience.models.aifs import AIFS


# ============================================================================
# Config loader
# ============================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================================
# ERA5 variable list builder  (matches train.py)
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


# ============================================================================
# Normalisation  (consistent with train.py)
# ============================================================================

def compute_normalisation_params(
    cfg: dict, variable_names: List[str],
    statistics_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-variable norm_mul and norm_add from config.

    If ``statistics_path`` provides a ``.npz`` with ``mean``/``stdev``
    arrays, those values are used.  Otherwise identity.
    """
    nc = cfg["normalizer"]
    default_method = nc["default"]
    remap = nc.get("remap", {}) or {}
    methods: Dict[str, str] = {}
    for v in variable_names:
        if v in (nc.get("none") or []):
            methods[v] = "none"
        elif v in (nc.get("max") or []):
            methods[v] = "max"
        elif v in (nc.get("min-max") or []):
            methods[v] = "min-max"
        elif v in (nc.get("std") or []):
            methods[v] = "std"
        else:
            methods[v] = default_method

    name_to_idx = {n: i for i, n in enumerate(variable_names)}
    num_vars = len(variable_names)
    _mean = np.zeros(num_vars, dtype=np.float32)
    _stdev = np.ones(num_vars, dtype=np.float32)

    if statistics_path and os.path.exists(statistics_path):
        stats = np.load(statistics_path)
        if "variables" in stats:
            ds_vars = [str(v) for v in stats["variables"]]
        else:
            import json
            import onescience.models.aifs as aifs_pkg
            cfg_json_path = os.path.join(os.path.dirname(aifs_pkg.__file__), "aifs_config.json")
            ds_vars = json.load(
                open(cfg_json_path)
            )["dataset"]["variables"]
        ds_name_to_idx = {n: i for i, n in enumerate(ds_vars)}
        for name, i in name_to_idx.items():
            if name in ds_name_to_idx:
                j = ds_name_to_idx[name]
                _mean[i] = float(stats["mean"][j])
                _stdev[i] = float(stats["stdev"][j])
        print(f"[INFO] Loaded statistics from {statistics_path}")

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


def build_denorm_params(norm_mul: np.ndarray, norm_add: np.ndarray):
    """Invert normalisation:  x = (x_norm - add) / mul."""
    eps = 1e-9
    denorm_mul = np.where(np.abs(norm_mul) > eps, 1.0 / norm_mul, 1.0)
    denorm_add = -norm_add * denorm_mul
    return denorm_mul.astype(np.float32), denorm_add.astype(np.float32)


# ============================================================================
# Forcing features  (consistent with train.py)
# ============================================================================

def _compute_insolation(
    ts: datetime.datetime, lat_rad: np.ndarray, lon_rad: np.ndarray
) -> np.ndarray:
    ref = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=pytz.utc)
    mt = ts.timestamp()
    days = (mt - ref.timestamp()) / (24.0 * 3600.0)
    jc = days / 36525.0
    theta = 67310.54841 + jc * (
        876600.0 * 3600.0
        + 8640184.812866
        + jc * (0.093104 - jc * 6.2e-5)
    )
    gmst = np.fmod((theta / 240.0) * np.pi / 180.0, 2.0 * np.pi)
    ma = np.deg2rad(
        357.52910 + 35999.05030 * jc - 0.0001559 * jc**2 - 0.00000048 * jc**3
    )
    ml = np.deg2rad(280.46645 + 36000.76983 * jc + 0.0003032 * jc**2)
    dl = np.deg2rad(
        (1.914600 - 0.004817 * jc - 0.000014 * jc**2) * np.sin(ma)
        + (0.019993 - 0.000101 * jc) * np.sin(2.0 * ma)
        + 0.000290 * np.sin(3.0 * ma)
    )
    tl = ml + dl
    eps = np.deg2rad(
        23.0
        + 26.0 / 60.0
        + 21.406 / 3600.0
        - (
            46.836769 * jc
            - 0.0001831 * jc**2
            + 0.00200340 * jc**3
            - 0.576e-6 * jc**4
            - 4.34e-8 * jc**5
        )
        / 3600.0
    )
    x = np.cos(tl)
    y = np.cos(eps) * np.sin(tl)
    z = np.sin(eps) * np.sin(tl)
    r = np.sqrt(1.0 - z * z)
    dec = np.arctan2(z, r)
    ra = 2.0 * np.arctan2(y, x + r)
    ha = (gmst + lon_rad) - ra
    cos_z = np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.cos(ha)
    return cos_z.astype(np.float32)


def compute_forcing_features(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    timestamps: List[datetime.datetime],
    static_fields: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute the 13 forcing channels for a 2-timestep window.

    *static_fields* supplies lsm, z, slor, sdor (time-invariant).
    """
    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)
    multi_step = len(timestamps)
    forcing: Dict[str, np.ndarray] = {}

    forcing["cos_latitude"] = np.tile(
        np.cos(lat_rad)[np.newaxis, :], (multi_step, 1)
    )
    forcing["sin_latitude"] = np.tile(
        np.sin(lat_rad)[np.newaxis, :], (multi_step, 1)
    )
    forcing["cos_longitude"] = np.tile(
        np.cos(lon_rad)[np.newaxis, :], (multi_step, 1)
    )
    forcing["sin_longitude"] = np.tile(
        np.sin(lon_rad)[np.newaxis, :], (multi_step, 1)
    )

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

    # Static fields — copied from initial conditions, held invariant
    for var_name in ["lsm", "z", "slor", "sdor"]:
        if var_name in static_fields:
            forcing[var_name] = np.tile(
                static_fields[var_name][:1], (multi_step, 1)
            )
    return forcing


# ============================================================================
# N320 interpolation
# ============================================================================

def _interp_n320(arr: np.ndarray) -> np.ndarray:
    return ekr.interpolate(arr, {"grid": (0.25, 0.25)}, {"grid": "N320"})


# ============================================================================
# ERA5 frame loading
# ============================================================================

def load_frames(era5_dir: str, year: int, step_idx: int, used_vars: List[str]):
    """Load two consecutive frames for the initial condition."""
    # ERA5Dataset expects dataset_dir to contain data/{year}.h5,
    # so if our path is ./fake_era5, data is at ./fake_era5/data/2008.h5
    ds = ERA5Dataset(
        dataset_dir=era5_dir,
        used_years=[year],
        used_variables=used_vars,
        input_steps=2,
        output_steps=0,
        normalize=False,
    )
    invar, _, _, _, _ = ds[step_idx]
    return invar[0].numpy(), invar[1].numpy()


# ============================================================================
# ERA5 → AIFS field construction
# ============================================================================

def build_aifs_fields_from_frames(
    frame_tm6: np.ndarray,
    frame_t0: np.ndarray,
    era5_var_list: List[str],
    cfg: dict,
    diag_map: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    """Convert raw ERA5 frames to per-variable N320-interpolated fields.

    Returns
    -------
    dict  {aifs_name: np.array([frame_t-6h, frame_t0])}
    """
    av = cfg["aifs_variables"]
    em = cfg["era5_mapping"]
    name_to_ch = {n: i for i, n in enumerate(era5_var_list)}
    fields: Dict[str, np.ndarray] = {}

    for aifs_name in av["surface"]:
        ch = name_to_ch[em["surface"][aifs_name]]
        fields[aifs_name] = np.stack(
            [_interp_n320(frame_tm6[ch]), _interp_n320(frame_t0[ch])]
        )
    for aifs_name in av["soil"]:
        ch = name_to_ch[em["soil"][aifs_name]]
        fields[aifs_name] = np.stack(
            [_interp_n320(frame_tm6[ch]), _interp_n320(frame_t0[ch])]
        )
    for aifs_var in av["pressure_level"]:
        tpl = em["pressure_level"][aifs_var]
        for level in av["pressure_levels"]:
            ch = name_to_ch[tpl.format(level=level)]
            fields[f"{aifs_var}_{level}"] = np.stack(
                [_interp_n320(frame_tm6[ch]), _interp_n320(frame_t0[ch])]
            )
    _diag_map = diag_map if diag_map is not None else em["diagnostic"]
    for aifs_name in av["diagnostic"]:
        era5_name = _diag_map.get(aifs_name)
        if era5_name and era5_name in name_to_ch:
            ch = name_to_ch[era5_name]
            fields[aifs_name] = np.stack(
                [_interp_n320(frame_tm6[ch]), _interp_n320(frame_t0[ch])]
            )
    return fields


# ============================================================================
# Autoregressive forecaster
# ============================================================================

class AIFSForecaster:
    """Deterministic autoregressive forecaster using the AIFS model.

    Parameters
    ----------
    model : AIFS
        Loaded AIFS model wrapper.
    norm_mul : np.ndarray  shape (V_in,)
        Per-input-channel multiplicative normalisation.
    norm_add : np.ndarray  shape (V_in,)
        Per-input-channel additive normalisation.
    denorm_mul : np.ndarray  shape (V_out,)
        Per-output-channel multiplicative denormalisation.
    denorm_add : np.ndarray  shape (V_out,)
        Per-output-channel additive denormalisation.
    device : str
        PyTorch device.
    """

    # Variables that are time-invariant — copied from initial condition
    _STATIC_NAMES = {"lsm", "z", "slor", "sdor"}

    def __init__(
        self,
        model: AIFS,
        norm_mul: np.ndarray,
        norm_add: np.ndarray,
        denorm_mul: np.ndarray,
        denorm_add: np.ndarray,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device

        self._input_vars = model.input_variables
        self._output_vars = model.output_variables
        self._norm_mul = norm_mul.astype(np.float32)
        self._norm_add = norm_add.astype(np.float32)
        self._denorm_mul = denorm_mul.astype(np.float32)
        self._denorm_add = denorm_add.astype(np.float32)

        # Channel lookups
        self._in_ch: Dict[str, int] = {
            n: i for i, n in enumerate(self._input_vars)
        }
        self._out_ch: Dict[str, int] = {
            n: i for i, n in enumerate(self._output_vars)
        }

        # Input variable → output variable channel mapping (for feedback)
        # Only prognostic vars exist in both input and output.
        self._in_to_out: Dict[int, int] = {}
        for i_idx, name in enumerate(self._input_vars):
            if name in self._out_ch:
                self._in_to_out[i_idx] = self._out_ch[name]

        # Identify which input vars are prognostic (fed back from output)
        # vs forcing (computed each step).
        self._prognostic_input_indices = sorted(self._in_to_out.keys())
        self._forcing_input_indices = sorted(
            set(range(len(self._input_vars))) - set(self._in_to_out.keys())
        )

        n_prog, n_forc = (
            len(self._prognostic_input_indices),
            len(self._forcing_input_indices),
        )
        print(
            f"[INFO] Forecaster: {n_prog} prognostic + {n_forc} forcing "
            f"= {len(self._input_vars)} input vars → {len(self._output_vars)} output vars"
        )

    # ------------------------------------------------------------------
    def forecast(
        self,
        initial_fields: Dict[str, np.ndarray],
        start_date: datetime.datetime,
        lead_time_hours: int,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> List[Dict]:
        """Run an autoregressive forecast.

        Parameters
        ----------
        initial_fields : dict  {aifs_name: np.array([frame_t-6h, frame_t0])}
            Initial atmospheric state on the N320 grid.
        start_date : datetime
            Valid time of the second frame (t0).
        lead_time_hours : int
            Total forecast length in hours (multiple of 6).
        latitudes, longitudes : np.ndarray
            N320 node coordinates.

        Returns
        -------
        list of dict
            One dict per output step::
                {"date": datetime, "latitudes": array, "longitudes": array,
                 "fields": {var_name: array}}
        """
        num_steps = lead_time_hours // 6
        states: List[Dict] = []
        fields = initial_fields.copy()
        current_date = start_date
        num_nodes = len(latitudes)

        t_start = time.time()
        pbar = tqdm(
            range(num_steps), desc="Forecast", unit="step",
            dynamic_ncols=True,
        )
        for step in pbar:
            # ---- 1.  Timestamps for this window ----------------------------
            t0 = current_date
            t_m6 = t0 - datetime.timedelta(hours=6)

            # ---- 2.  Compute time-dependent forcing -----------------------
            forcing = compute_forcing_features(
                latitudes, longitudes, [t_m6, t0], fields
            )

            # ---- 3.  Assemble normalised input tensor (2, G, V_in) --------
            x = np.zeros(
                (2, num_nodes, len(self._input_vars)), dtype=np.float32
            )
            for var_name, ch in self._in_ch.items():
                if var_name in fields:
                    x[:, :, ch] = fields[var_name]
                elif var_name in forcing:
                    x[:, :, ch] = forcing[var_name]
                # else: stays zero (should not happen for valid inputs)

            x = x * self._norm_mul[np.newaxis, np.newaxis, :] + self._norm_add[
                np.newaxis, np.newaxis, :
            ]

            # ---- 4.  Model forward ----------------------------------------
            x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model.predict(x_t)  # (1, G, V_out)
            pred_np = pred[0].cpu().numpy()  # (G, V_out)

            # ---- 5.  Denormalise output -----------------------------------
            pred_phys = (
                pred_np * self._denorm_mul[np.newaxis, :]
                + self._denorm_add[np.newaxis, :]
            )

            # ---- 6.  Parse into fields dict -------------------------------
            next_fields: Dict[str, np.ndarray] = {}
            for var_name, ch in self._out_ch.items():
                next_fields[var_name] = pred_phys[:, ch].copy()

            # ---- 7.  Save state -------------------------------------------
            next_date = t0 + datetime.timedelta(hours=6)
            states.append(
                {
                    "date": next_date,
                    "latitudes": latitudes.copy(),
                    "longitudes": longitudes.copy(),
                    "fields": {k: v.copy() for k, v in next_fields.items()},
                }
            )

            # ---- 8.  Shift for next autoregressive step -------------------
            # New window: [old t0, prediction]
            shifted: Dict[str, np.ndarray] = {}
            for var_name in fields:
                if var_name in next_fields:
                    shifted[var_name] = np.stack(
                        [fields[var_name][1], next_fields[var_name]], axis=0
                    )
                elif var_name in self._STATIC_NAMES:
                    # Static fields: keep invariant
                    shifted[var_name] = fields[var_name].copy()
            fields = shifted
            current_date = next_date

            # 更新进度条
            elapsed = time.time() - t_start
            pbar.set_postfix({
                "date": next_date.strftime("%m-%d %H:%M"),
                "spd": f"{elapsed / (step + 1):.1f}s",
            })

        elapsed = time.time() - t_start
        print(
            f"[INFO] Forecast done: {num_steps} steps in {elapsed:.1f}s "
            f"({elapsed / max(num_steps, 1):.2f}s/step)"
        )
        return states


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="AIFS v1.1 — 10-day Inference")
    p.add_argument("-c", "--config", default=str(ROOT / "conf" / "config.yaml"))
    p.add_argument("--checkpoint", default=None, help="override checkpoint path")
    p.add_argument("--year", type=int, default=None, help="override test year")
    p.add_argument(
        "--lead_time", type=int, default=None, help="override lead time (h)"
    )
    p.add_argument("-o", "--output_dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--device_ids", default=None)
    args = p.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    hw = cfg.get("hardware", {})
    ck = cfg.get("checkpoint", {})
    data = cfg.get("data", {})
    out = cfg.get("output", {})

    device_str = args.device or hw.get("device", "dcu")
    device_ids = args.device_ids or str(hw.get("device_ids", "0"))
    ckpt = args.checkpoint or ck.get("pretrained", "")
    test_years = data.get("test_years", [2008])
    test_year = args.year or test_years[0]
    lead_time = args.lead_time or data.get("test_lead_time", 240)
    era5_dir = data.get("data_dir", str(ROOT / "data" / "fake_era5"))
    out_dir = args.output_dir or str(ROOT / "output")

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    if device_str == "dcu":
        os.environ["HIP_VISIBLE_DEVICES"] = device_ids
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cpu":
        print("warning: CPU running")
        device = "cpu"

    # ---- Analysis date: Jan 1 06:00 of test_year -------------------------
    step_idx = 1  # second 6h step → Jan 1 06:00
    analysis_date = datetime.datetime(
        test_year, 1, 1, 6, 0, 0, tzinfo=pytz.utc
    )
    date_str = analysis_date.strftime("%Y%m%d%H")

    print(f"\n{'='*60}\n  AIFS v1.1 — 10-day Forecast\n{'='*60}")
    print(f"  Test year : {test_year}  |  analysis: {date_str}")
    print(
        f"  Lead time : {lead_time}h ({lead_time // 6} steps)"
    )
    print(f"  Device    : {device}  |  ckpt: {ckpt}\n{'='*60}\n")

    try:
        # ---- 1.  Load AIFS model ----------------------------------------
        t0 = time.time()
        model = AIFS(ckpt, device=device, pretrained=True)
        print(f"[INFO] Model loaded ({time.time() - t0:.1f}s)")

        input_vars = model.input_variables
        output_vars = model.output_variables
        grid_lat = model.latitudes.cpu().numpy()
        grid_lon = model.longitudes.cpu().numpy()

        # ---- 2.  Compute normalisation params ---------------------------
        stats_path = cfg["normalizer"].get("statistics_path") or None
        norm_mul_in, norm_add_in = compute_normalisation_params(
            cfg, input_vars, statistics_path=stats_path,
        )
        norm_mul_out, norm_add_out = compute_normalisation_params(
            cfg, output_vars, statistics_path=stats_path,
        )
        denorm_mul, denorm_add = build_denorm_params(norm_mul_out, norm_add_out)

        # ---- 3.  Load ERA5 initial frames -------------------------------
        era5_list = build_era5_variable_list(cfg)

        # Auto-correct names against available H5 variables
        data_dir = os.path.join(era5_dir, "data")
        if os.path.isdir(data_dir):
            h5_files = sorted(
                [f for f in os.listdir(data_dir) if f.endswith(".h5")]
            )
            available_set: set = set()
            if h5_files:
                import h5py

                with h5py.File(
                    os.path.join(data_dir, h5_files[0]), "r"
                ) as f:
                    available_vars = [
                        v.decode() if isinstance(v, bytes) else v
                        for v in f["fields"].attrs["variables"]
                    ]
                available_set = set(available_vars)
            corrected = _auto_correct_variable_names(
                era5_list, available_set, cfg
            )
            used_vars = [v for v in corrected if v in available_set]
            missing = [v for v in corrected if v not in available_set]
            if missing:
                print(
                    f"[WARN] {len(missing)} vars missing from H5: {missing[:10]}"
                )
        else:
            used_vars = era5_list

        # Build diagnostic map with corrected names
        diag_map: Dict[str, str] = {}
        for aifs_name, orig in cfg["era5_mapping"]["diagnostic"].items():
            cn = _auto_correct_variable_names([orig], available_set, cfg)[0]
            diag_map[aifs_name] = (
                cn if cn in available_set else orig
            )

        ft6, ft0 = load_frames(era5_dir, test_year, step_idx, used_vars)
        fields = build_aifs_fields_from_frames(
            ft6, ft0, used_vars, cfg, diag_map=diag_map
        )

        # ---- 4.  Autoregressive forecast --------------------------------
        forecaster = AIFSForecaster(
            model,
            norm_mul_in,
            norm_add_in,
            denorm_mul,
            denorm_add,
            device=device,
        )

        initial_state = {
            "date": analysis_date,
            "fields": fields,
        }

        states = forecaster.forecast(
            initial_fields=fields,
            start_date=analysis_date,
            lead_time_hours=lead_time,
            latitudes=grid_lat,
            longitudes=grid_lon,
        )

        # ---- 5.  Save to .npz -------------------------------------------
        os.makedirs(out_dir, exist_ok=True)
        for s in states:
            vt = s["date"].strftime("%Y%m%d%H")
            path = os.path.join(out_dir, f"aifs_forecast_{vt}.npz")
            d = dict(
                latitudes=s["latitudes"], longitudes=s["longitudes"]
            )
            d.update(s["fields"])
            np.savez_compressed(
                path, **d, date=np.array(str(s["date"]))
            )
            print(f"[INFO] Saved: {path}")

        print(f"[INFO] Done — {len(states)} files → {out_dir}")

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
