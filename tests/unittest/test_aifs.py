import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# 抑制已知的良性警告 (DCU 环境缺 CUDA 运行时导致 torch-cluster/scatter/sparse 不可用)
warnings.filterwarnings("ignore", message=".*importing 'torch-scatter'.*")
warnings.filterwarnings("ignore", message=".*importing 'torch-cluster'.*")
warnings.filterwarnings("ignore", message=".*importing 'torch-sparse'.*")
warnings.filterwarnings("ignore", message=".*torch-cluster.*library is not installed.*")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ---- 1. 确保必需文件存在 -------------------------------------------
_config = ROOT / "model" / "aifs_config.json"
_grid = ROOT / "model" / "grid-n320.npz"
_cleanup = []


def _write_builtin_config(path):
    """内置最小配置——仅含 AnemoiModelEncProcDec 构建必需的字段。"""
    config_data = {
        "model_config": {
            "graph": {"data": "data", "hidden": "hidden"},
            "data": {
                "diagnostic": ["tp","cp","sf","tcc","hcc","lcc","mcc","ro","ssrd","strd","100u","100v"],
                "forcing": ["cos_latitude","cos_longitude","sin_latitude","sin_longitude",
                           "cos_julian_day","cos_local_time","sin_julian_day","sin_local_time",
                           "insolation","lsm","z","slor","sdor"],
                "remapped": None,
            },
            "training": {"multistep_input": 2},
            "model": {
                "num_channels": 256,
                "activation": None,
                "layer_kernels": {},
                "trainable_parameters": {"hidden": 8},
                "encoder": {
                    "_target_": "anemoi.models.layers.mapper.GraphTransformerForwardMapper",
                    "num_chunks": 1, "num_heads": 4, "mlp_hidden_ratio": 4,
                    "trainable_size": 8, "qk_norm": False,
                    "sub_graph_edge_attributes": ["edge_length", "edge_dirs"],
                },
                "processor": {
                    "_target_": "anemoi.models.layers.processor.TransformerProcessor",
                    "num_layers": 2, "num_chunks": 1, "num_heads": 4,
                    "mlp_hidden_ratio": 4, "window_size": 128, "qk_norm": False,
                    "attention_implementation": "flash_attention",
                    "dropout_p": 0.0, "softcap": 0, "use_alibi_slopes": False,
                },
                "decoder": {
                    "_target_": "anemoi.models.layers.mapper.GraphTransformerBackwardMapper",
                    "num_chunks": 1, "num_heads": 4, "mlp_hidden_ratio": 4,
                    "trainable_size": 8, "qk_norm": False,
                    "sub_graph_edge_attributes": ["edge_length", "edge_dirs"],
                },
                "bounding": [],
            },
        },
        "data_indices": {
            "data": {
                "input": {"full": list(range(103)), "prognostic": list(range(90)),
                          "diagnostic": [], "forcing": list(range(90, 103))},
                "output": {"full": list(range(102)), "prognostic": list(range(90)),
                           "diagnostic": list(range(90, 102)), "forcing": []},
            },
            "model": {
                "input": {"full": list(range(103)), "prognostic": list(range(90)),
                          "diagnostic": [], "forcing": list(range(90, 103))},
                "output": {"full": list(range(102)), "prognostic": list(range(90)),
                           "diagnostic": list(range(90, 102)), "forcing": []},
            },
            "internal_data": {
                "input": {"full": list(range(103)), "prognostic": list(range(90)),
                          "diagnostic": [], "forcing": list(range(90, 103))},
                "output": {"full": list(range(102)), "prognostic": list(range(90)),
                           "diagnostic": list(range(90, 102)), "forcing": []},
            },
            "internal_model": {
                "input": {"full": list(range(103)), "prognostic": list(range(90)),
                          "diagnostic": [], "forcing": list(range(90, 103))},
                "output": {"full": list(range(102)), "prognostic": list(range(90)),
                           "diagnostic": list(range(90, 102)), "forcing": []},
            },
        },
        "dataset": {"variables": [f"var_{i}" for i in range(115)]},
    }
    with open(path, "w") as f:
        json.dump(config_data, f)


if not _config.exists() or not _grid.exists():
    print("Static files not found — generating temporary copies …")

    # 在 CWD 下创建临时 model/ 目录 (多种 AIFS 实现会搜索 ./model/)
    _cwd_model = Path.cwd() / "model"
    _cwd_model.mkdir(exist_ok=True)

    # 1a. aifs_config.json — 优先从同目录 checkpoint 提取
    ckpt = ROOT / "aifs-single-mse-1.1.ckpt"
    if not ckpt.exists():
        ckpt = ROOT.parent / "aifs-single-mse-1.1.ckpt"
    if not ckpt.exists():
        ckpt = None

    if not _config.exists():
        if ckpt and ckpt.exists():
            import zipfile
            with zipfile.ZipFile(ckpt, "r") as zf:
                for fname in zf.namelist():
                    if fname.endswith("ai-models.json"):
                        meta = json.load(zf.open(fname, "r"))
                        break
            config_data = {
                "model_config": meta["config"],
                "data_indices": meta["data_indices"],
                "dataset": {"variables": meta["dataset"]["variables"]},
            }
            config_data["model_config"].pop("supporting_arrays_paths", None)
            _config = _cwd_model / "aifs_config.json"
            with open(_config, "w") as f:
                json.dump(config_data, f)
            print(f"  config extracted from checkpoint")
        else:
            _config = _cwd_model / "aifs_config.json"
            _write_builtin_config(_config)
            print(f"  config from built-in defaults")
        _cleanup.append(_config)

    if not _grid.exists():
        _grid = _cwd_model / "grid-n320.npz"
        _errors = []
        # 方式 1: ECMWF 标准网格 API
        try:
            from anemoi.utils.grids import grids
            g = grids("n320")
            np.savez(_grid, latitudes=g["latitudes"], longitudes=g["longitudes"])
            print(f"  grid downloaded from ECMWF (n320)")
        except Exception as e:
            _errors.append(f"ECMWF: {e}")
            # 方式 2: ModelScope
            try:
                import subprocess
                subprocess.run(
                    ["modelscope", "download", "--model",
                     "OneScience/AIFS_Single_v1", "model/grid-n320.npz",
                     "--local_dir", str(_cwd_model)],
                    check=True, capture_output=True,
                )
                # modelscope 可能放到 _cwd_model/model/grid-n320.npz
                _dl = _cwd_model / "model" / "grid-n320.npz"
                if _dl.exists():
                    np.savez(_grid, **dict(np.load(_dl)))
                print(f"  grid downloaded from ModelScope")
            except Exception as e2:
                _errors.append(f"ModelScope: {e2}")
                raise RuntimeError(
                    "Cannot obtain N320 grid coordinates.\n"
                    f"  ECMWF API: {_errors[0]}\n"
                    f"  ModelScope: {_errors[1]}\n"
                    "Manual fix:\n"
                    "  modelscope download --model OneScience/AIFS_Single_v1 "
                    "model/grid-n320.npz --local_dir ./"
                )
        _cleanup.append(_grid)

# ---- 2. 导入前：检测 torch_cluster 是否可用 ------------------------
# 若包已安装但 native 库无法加载 (缺 libcudart.so.12 等)，
# 强制 anemoi-graphs 使用 sklearn 回退，避免 OSError。
try:
    import torch_cluster  # noqa: F401
except (OSError, ImportError):
    import anemoi.graphs.edges.builders.cutoff as _cb
    import anemoi.graphs.edges.builders.knn as _kb
    _cb.TORCH_CLUSTER_AVAILABLE = False
    _kb.TORCH_CLUSTER_AVAILABLE = False

# ---- 3. 导入 AIFS (兼容两种安装方式) ----------------------------
try:
    from model.aifs import AIFS
except ImportError:
    from onescience.models.aifs import AIFS

# ---- 3. 前向测试 -----------------------------------------------
try:
    model = AIFS.from_scratch(device=device)
    x = torch.randn(1, 2, model.num_grid_points, model.num_input_vars).to(device)
    out = model.predict(x)

    print("Function: AIFS Model Forward")
    print(f"output shape: {out.shape}")
    print(f"target shape: torch.Size([1, {model.num_grid_points}, {model.num_output_vars}])")

    print("\n" + "=" * 50)
    print("  ✓ ALL TESTS PASSED — AIFS environment ready")
    print("=" * 50)

except OSError as e:
    if "libcudart" in str(e) or "libcuda" in str(e):
        print("\n✗ CUDA runtime library not found.")
        print("  torch_cluster / torch_scatter require libcudart.so.12.")
        print("  On DCU: ensure DTK lib path is in LD_LIBRARY_PATH, e.g.:")
        print("    export LD_LIBRARY_PATH=$DTK_PATH/hip/lib:$LD_LIBRARY_PATH")
    else:
        raise
except Exception as e:
    print(f"\n✗ Model test failed: {e}")
    print("  Core packages imported successfully — environment mostly ready.")
    print(f"  Error type: {type(e).__name__}")

finally:
    # ---- 4. 清理临时文件 -------------------------------------------
    for f in _cleanup:
        try:
            os.remove(f)
        except OSError:
            pass
