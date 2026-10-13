import os
import h5py
import numpy as np
import xarray as xr
from onescience.utils.YParams import YParams


# 各数据集固定的空间和时间维度
DATASET_DIMS = {"T": 10, "H": 721, "W": 1440, "time_step": 6}
PATCH_HW = 32
PATCHES_PER_STEP = 2
RNG_SEED = 42

def generate_fake_h5(data_dir, var_names, years, dims):
    """
    为每个年份生成一个稀疏随机 h5 文件。
    仅写入少量随机 patch，其余 chunk 仍保持 fill_value=0，
    这样既能让 time-diff std 非零，也能保持文件很小。
    """
    os.makedirs(os.path.join(data_dir, "data"), exist_ok=True)
    T, C = dims["T"], len(var_names)
    H, W = dims["H"], dims["W"]
    patch_h = min(PATCH_HW, H)
    patch_w = min(PATCH_HW, W)

    means = np.zeros((1, C, 1, 1), dtype=np.float32)
    stds  = np.ones((1, C, 1, 1), dtype=np.float32)

    for year_idx, year in enumerate(years):
        path = os.path.join(data_dir, "data", f"{year}.h5")
        rng = np.random.default_rng(RNG_SEED + year_idx)
        with h5py.File(path, "w") as f:
            ds = f.create_dataset(
                "fields",
                shape=(T, C, H, W),
                dtype="float32",
                chunks=(1, C, patch_h, patch_w),
                fillvalue=0.0,
            )
            ds.attrs["variables"] = var_names
            ds.attrs["time_step"] = dims["time_step"]

            f.create_dataset("global_means", data=means)
            f.create_dataset("global_stds",  data=stds)

            for t in range(T):
                for _ in range(PATCHES_PER_STEP):
                    top = int(rng.integers(0, H - patch_h + 1))
                    left = int(rng.integers(0, W - patch_w + 1))
                    patch = rng.standard_normal((C, patch_h, patch_w)).astype(np.float32)
                    patch += np.float32(0.1 * t)
                    ds[t, :, top:top + patch_h, left:left + patch_w] = patch

        size_kb = os.path.getsize(path) / 1024
        print(f"  {year}.h5  shape=({T},{C},{H},{W})  "
              f"logical={T*C*H*W*4/1024**3:.1f}GB  actual={size_kb:.1f}KB")


def get_static(data_dir, var, name):
    os.makedirs(data_dir, exist_ok=True)
    ds = xr.Dataset(
        data_vars={
            f"{var}": (("valid_time", "latitude", "longitude"),
                np.random.rand(1, 721, 1440).astype(np.float32))
        },
        coords={
            "valid_time": ["2015-12-31"],
            "latitude": np.linspace(90, -90, 721, dtype=np.float64),
            "longitude": np.linspace(0, 359.75, 1440, dtype=np.float64),
            "number": 0,
            "expver": "",
        },
        attrs={
            "GRIB_centre": "ecmf",
            "GRIB_centreDescription": "European Centre for Medium-Range Weather Forecasts",
            "GRIB_subCentre": "0",
            "Conventions": "CF-1.7",
            "institution": "European Centre for Medium-Range Weather Forecasts",
            "history": "Generated manually",
        }
    )

    ds.to_netcdf(f"{data_dir}/{name}.nc")
    arr = np.random.randn(721, 1440).astype(np.float32)
    np.save(f'{data_dir}/land_mask.npy', arr)
    np.save(f'{data_dir}/soil_type.npy', arr)
    np.save(f'{data_dir}/topography.npy', arr)
    print(f"✅ Static data: {arr.shape}, dtype: {arr.dtype}, save to {data_dir}")


if __name__ == "__main__":
    cfg_datapipe = YParams("conf/config.yaml", "datapipe")

    if cfg_datapipe.dataset.data_dir.startswith("/public/") or cfg_datapipe.dataset.data_dir.startswith("/work2/"):
        print("请检查 config，确保各 *_dir 指向本地测试路径而非生产路径。")
        exit()

    years    = cfg_datapipe.dataset.train_time + cfg_datapipe.dataset.val_time + cfg_datapipe.dataset.test_time
    atm_vars = cfg_datapipe.dataset.channels

    generate_fake_h5(cfg_datapipe.dataset.data_dir, atm_vars, years, DATASET_DIMS)

    static_dir = os.path.join(cfg_datapipe.dataset.data_dir, "static")
    get_static(static_dir, 'z', 'geopotential')
    get_static(static_dir, 'lsm', 'land_sea_mask')


    print("\n✅ Fake datasets generated.")
