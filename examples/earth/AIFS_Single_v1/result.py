#!/usr/bin/env python
# coding: utf-8
"""
AIFS v1.1 — Evaluation & Visualisation
========================================
Computes RMSE and ACC (Anomaly Correlation Coefficient) on inference
results, then plots selected variables on the N320 Gaussian grid.

Usage:
    python result.py
    python result.py -v 2t,z_500,tp
    python result.py -v all
    python result.py --no-metrics           # plot only
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ============================================================================
# Variable metadata
# ============================================================================
_VAR_META = {
    "10u": ("10m U-wind", "RdBu_r", "m/s"),
    "10v": ("10m V-wind", "RdBu_r", "m/s"),
    "2t": ("2m Temperature", "RdYlBu_r", "K"),
    "2d": ("2m Dewpoint", "RdYlBu_r", "K"),
    "msl": ("MSLP", "RdYlBu_r", "Pa"),
    "skt": ("Skin Temp", "RdYlBu_r", "K"),
    "sp": ("Surface Pressure", "RdYlBu_r", "Pa"),
    "tcw": ("Total Column Water", "Blues", "kg/m²"),
    "z": ("Surface Geopotential", "terrain", "m²/s²"),
    "tp": ("Total Precip", "YlGnBu", "m"),
    "cp": ("Convective Precip", "YlGnBu", "m"),
    "sf": ("Snowfall", "PuBu", "m"),
    "tcc": ("Total Cloud Cover", "Greys", "0-1"),
    "lcc": ("Low Cloud", "Greys", "0-1"),
    "mcc": ("Mid Cloud", "Greys", "0-1"),
    "hcc": ("High Cloud", "Greys", "0-1"),
    "ro": ("Runoff", "YlGnBu", "m"),
    "ssrd": ("Solar Radiation", "YlOrRd", "J/m²"),
    "strd": ("Thermal Radiation", "YlOrRd", "J/m²"),
    "100u": ("100m U-wind", "RdBu_r", "m/s"),
    "100v": ("100m V-wind", "RdBu_r", "m/s"),
    "stl1": ("Soil Temp L1", "RdYlBu_r", "K"),
    "stl2": ("Soil Temp L2", "RdYlBu_r", "K"),
    "swvl1": ("Soil Moisture L1", "Blues", "m³/m³"),
    "swvl2": ("Soil Moisture L2", "Blues", "m³/m³"),
}
_PL_META = {
    "z": ("Geopotential", "RdYlBu_r", "m²/s²"),
    "t": ("Temperature", "RdYlBu_r", "K"),
    "u": ("U-wind", "RdBu_r", "m/s"),
    "v": ("V-wind", "RdBu_r", "m/s"),
    "w": ("Vertical Velocity", "RdBu_r", "Pa/s"),
    "q": ("Specific Humidity", "Blues", "kg/kg"),
}
for _a, (_desc, _cmap, _unit) in list(_PL_META.items()):
    for _l in [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]:
        _VAR_META[f"{_a}_{_l}"] = (f"{_desc} {_l}hPa", _cmap, _unit)

DEFAULT_VARS = ["10u", "2t", "msl", "tp", "z_500", "t_850"]


# ============================================================================
# Helpers
# ============================================================================

def _load_config():
    config_path = ROOT / "conf" / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(open(config_path))
    return {}

# ============================================================================
# ACC / RMSE computation
# ============================================================================

def compute_metrics(pred_dir: str, label_dir: str, test_years: list):
    """Compute per-variable RMSE and ACC vs ERA5 ground truth.

    Parameters
    ----------
    pred_dir : str
        Directory of ``aifs_forecast_*.npz`` files.
    label_dir : str
        Root of ERA5 H5 data (contains ``data/{year}.h5``).
    test_years : list[int]
        Years to evaluate.
    """
    import h5py

    files = sorted(glob.glob(os.path.join(pred_dir, "aifs_forecast_*.npz")))
    if not files:
        raise FileNotFoundError(f"No forecast .npz in {pred_dir}")

    # Discover variables from first prediction
    sample = np.load(files[0])
    var_names = sorted(
        k for k in sample.keys()
        if k not in ("latitudes", "longitudes", "date")
    )
    num_vars = len(var_names)
    sample.close()

    # Build AIFS → ERA5 H5 channel mapping from config
    cfg = _load_config()
    em = cfg.get("era5_mapping", {})
    av = cfg.get("aifs_variables", {})
    aifs_to_era5 = {}
    for cat in ["surface", "soil", "diagnostic"]:
        for aifs_name, era5_name in em.get(cat, {}).items():
            aifs_to_era5[aifs_name] = era5_name
    for vv in av.get("pressure_level", []):
        tpl = em.get("pressure_level", {}).get(vv, "")
        for lvl in av.get("pressure_levels", []):
            aifs_to_era5[f"{vv}_{lvl}"] = tpl.format(level=lvl)

    # Accumulators
    num_samples = 0
    numerator = np.zeros(num_vars, dtype=np.float64)
    pred_sq = np.zeros(num_vars, dtype=np.float64)
    label_sq = np.zeros(num_vars, dtype=np.float64)
    rmse_sum = np.zeros(num_vars, dtype=np.float64)

    pbar = tqdm(files, desc="Computing metrics", unit="file")
    for fp in pbar:
        data = np.load(fp)
        fname = os.path.splitext(os.path.basename(fp))[0]
        date_str = fname.replace("aifs_forecast_", "")
        year = int(date_str[:4])
        if year not in test_years:
            data.close()
            continue

        # Load corresponding ERA5 label
        h5_path = os.path.join(label_dir, "data", f"{year}.h5")
        if not os.path.exists(h5_path):
            data.close()
            continue

        with h5py.File(h5_path, "r") as hf:
            ds = hf["fields"]
            h5_vars = [v.decode() if isinstance(v, bytes) else v
                       for v in ds.attrs["variables"]]
            h5_time_step = int(ds.attrs.get("time_step", 6))
            # Parse date → time index
            from datetime import datetime
            dt = datetime.strptime(date_str, "%Y%m%d%H")
            year_start = datetime(dt.year, 1, 1)
            hours = (dt - year_start).total_seconds() / 3600
            t_idx = int(hours / h5_time_step)
            if t_idx >= ds.shape[0]:
                data.close()
                continue
            label_full = hf["fields"][t_idx]  # (C, 721, 1440)

        # Build per-file H5 channel index
        h5_ch_map = {}
        for vi, vname in enumerate(var_names):
            era5_name = aifs_to_era5.get(vname)
            if era5_name and era5_name in h5_vars:
                h5_ch_map[vi] = h5_vars.index(era5_name)

        if not h5_ch_map:
            data.close()
            continue

        # Regrid labels to N320
        label_n320 = {}
        for vi, h5_ch in h5_ch_map.items():
            arr_2d = label_full[h5_ch]
            arr_1d = _interp_to_n320(arr_2d)
            if arr_1d is not None:
                label_n320[vi] = arr_1d

        # Compute metrics per variable
        for vi, label_1d in label_n320.items():
            pred = data[var_names[vi]]

            # RMSE
            sq_err = (pred - label_1d) ** 2
            rmse_sum[vi] += np.sqrt(sq_err.mean())

            # ACC: anomaly correlation (anomaly = deviation from spatial mean)
            pred_anom = pred - pred.mean()
            label_anom = label_1d - label_1d.mean()

            numerator[vi] += (pred_anom * label_anom).sum()
            pred_sq[vi] += (pred_anom ** 2).sum()
            label_sq[vi] += (label_anom ** 2).sum()

        num_samples += 1
        data.close()

    if num_samples == 0:
        raise RuntimeError("No matching label data found")

    rmse = rmse_sum / num_samples
    denom = np.sqrt(pred_sq * label_sq)
    denom = np.where(denom > 1e-8, denom, 1.0)
    acc = numerator / denom

    return var_names, rmse, acc


def _interp_to_n320(field_2d: np.ndarray) -> np.ndarray | None:
    """Interpolate (721, 1440) → N320 (542080,)."""
    try:
        import earthkit.regrid as ekr
        result = ekr.interpolate(
            field_2d[np.newaxis, ...].astype(np.float64),
            {"grid": (0.25, 0.25)},
            {"grid": "N320"},
        )
        return result.flatten().astype(np.float64)
    except Exception:
        return None


# ============================================================================
# Display
# ============================================================================

def print_metrics(var_names, rmse, acc):
    w = 24
    print(f"\n┌{'─' * (w + 2)}┬{'─' * 14}┬{'─' * 14}┐")
    print(f"│ {'Variable':<{w}} │ {'RMSE':>12} │ {'ACC':>12} │")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┤")
    for i in range(len(var_names)):
        print(f"│ {var_names[i]:<{w}} │ {rmse[i]:>12.6f} │ {acc[i]:>12.6f} │")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┤")
    print(f"│ {'Average':<{w}} │ {np.mean(rmse):>12.6f} │ {np.mean(acc):>12.6f} │")
    print(f"└{'─' * (w + 2)}┴{'─' * 14}┴{'─' * 14}┘")


# ============================================================================
# Plotting
# ============================================================================

def plot_field(state, var_name, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeat
    import matplotlib.tri as tri

    lats = state.get("latitudes")
    lons = state.get("longitudes")
    values = state.get("fields", {}).get(var_name)
    if lats is None or values is None:
        return None

    desc, cmap, unit = _VAR_META.get(var_name, (var_name, "viridis", ""))

    fig, ax = plt.subplots(
        figsize=(11, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeat.BORDERS, linestyle=":", linewidth=0.3)
    ax.set_global()

    lons_adj = np.where(lons > 180, lons - 360, lons)
    t = tri.Triangulation(lons_adj, lats)
    c = ax.tricontourf(t, values, levels=20, transform=ccrs.PlateCarree(),
                       cmap=cmap)
    cb = fig.colorbar(c, ax=ax, orientation="vertical", shrink=0.7, pad=0.02)
    if unit:
        cb.set_label(unit)

    vs = str(state.get("date", "unknown")).replace("T", " ")[:16]
    plt.title(f"{desc}  |  valid {vs}", fontsize=12)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{var_name}_{vs.replace(' ', '_').replace(':','')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="AIFS v1.1 — Evaluate & Visualise")
    p.add_argument("-c", "--config",
                   default=str(ROOT / "conf" / "config.yaml"))
    p.add_argument("-i", "--input_dir", default=None)
    p.add_argument("-p", "--plot_dir", default=None)
    p.add_argument("-v", "--variables", default="",
                   help="comma-separated vars; empty=default, 'all'=all")
    p.add_argument("-l", "--list", action="store_true",
                   help="list available vars and exit")
    p.add_argument("--no-metrics", action="store_true",
                   help="skip ACC/RMSE (plot only)")
    args = p.parse_args()

    cfg = _load_config()
    out_cfg = cfg.get("output", {})
    data_cfg = cfg.get("data", {})

    in_dir = args.input_dir or str(ROOT / "output")
    plt_dir = args.plot_dir or str(ROOT / "plots")
    test_years = data_cfg.get("test_years", [2008])

    files = sorted(glob.glob(os.path.join(in_dir, "aifs_forecast_*.npz")))
    if not files:
        print(f"[ERROR] No aifs_forecast_*.npz in {in_dir}")
        sys.exit(1)

    s = np.load(files[0])
    avail = [k for k in s.keys()
             if k not in ("latitudes", "longitudes", "date")]
    s.close()

    if args.list:
        print(f"\nAvailable variables ({len(avail)}):")
        for v in avail:
            d, _, u = _VAR_META.get(v, (v, "", ""))
            print(f"  {v:12s}  {d}")
        return

    if args.variables == "":
        vars_ = [v for v in DEFAULT_VARS if v in avail]
    elif args.variables == "all":
        vars_ = avail
    else:
        vars_ = [v.strip() for v in args.variables.split(",")
                 if v.strip() in avail]

    # ---- ACC / RMSE ----
    if not args.no_metrics:
        label_dir = data_cfg.get("data_dir", "")
        if label_dir and os.path.isdir(os.path.join(label_dir, "data")):
            try:
                var_names, rmse, acc = compute_metrics(
                    in_dir, label_dir, test_years,
                )
                print_metrics(var_names, rmse, acc)
                # Save metrics
                metrics_dir = ROOT / "metrics"
                os.makedirs(str(metrics_dir), exist_ok=True)
                np.save(str(metrics_dir / "rmse.npy"), rmse)
                np.save(str(metrics_dir / "acc.npy"), acc)
                with open(str(metrics_dir / "metrics.txt"), "w") as f:
                    f.write(f"{'Variable':<24s} {'RMSE':>12s} {'ACC':>12s}\n")
                    for i, v in enumerate(var_names):
                        f.write(f"{v:<24s} {rmse[i]:>12.6f} {acc[i]:>12.6f}\n")
                print(f"[INFO] Metrics saved to metrics/")
            except Exception as e:
                print(f"[WARN] ACC/RMSE skipped: {e}")
        else:
            print("[INFO] No label data found — skipping ACC/RMSE")

    # ---- Plotting ----
    print(f"[INFO] {len(files)} file(s), {len(vars_)} var(s): {', '.join(vars_[:6])}")
    total = 0
    for fp in files:
        d = np.load(fp)
        st = dict(
            date=str(d.get("date", "unknown")),
            latitudes=d["latitudes"],
            longitudes=d["longitudes"],
            fields={k: d[k] for k in avail if k in d},
        )
        for v in vars_:
            out_path = plot_field(st, v, plt_dir)
            if out_path:
                total += 1
        d.close()

    if total > 0:
        print(f"[INFO] Done: {total} plot(s) → {plt_dir}")
        # Print only first few to avoid flooding
    else:
        print("[WARN] No plots generated")


if __name__ == "__main__":
    main()
