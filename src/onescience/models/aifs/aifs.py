#!/usr/bin/env python
# coding: utf-8
"""
AIFS v1.1 — Onescience-style Model  (真正从零构建)
====================================================

``class AIFS(nn.Module)`` — 兼容 onescience 的 ECMWF AIFS
编码器-处理器-解码器图神经网络。

两种构建模式
------------
1. **从零构建** — 无需 ``.ckpt`` 文件，与 FengWu/Fuxi 一致::

       model = AIFS.from_scratch(device="cuda")

   从项目内的静态文件加载架构配置（``aifs_config.json``, 47 KB）和
   N320 网格坐标（``grid-n320.npz``, 4.2 MB），通过 ``anemoi-graphs``
   构建 HeteroData 图，由 ``AnemoiModelEncProcDec`` 随机初始化所有权重。

2. **加载预训练** — 向后兼容::

       model = AIFS("aifs.ckpt", device="cuda", pretrained=True)

架构 (anemoi-models 0.5.0)
--------------------------
- **Encoder**  : ``GraphTransformerForwardMapper`` — data nodes → hidden nodes
- **Processor**: ``TransformerProcessor`` × 16 layers — 在 o96 隐藏图上做
  滑动窗口注意力 (window_size=1120, flash_attention, 16 heads)
- **Decoder**  : ``GraphTransformerBackwardMapper`` — hidden → data nodes
- **Bounding** : ``ReLUBounding`` / ``HardtanhBounding`` / ``FractionBounding``
  — 强制输出变量满足物理约束

静态文件说明
------------
- ``aifs_config.json``: 从官方 checkpoint 的 ``ai-models.json`` 提取
  (model_config + data_indices + dataset), 定义完整模型架构
- ``grid-n320.npz``: N320 高斯网格经纬度坐标 (542,080 节点), 从官方
  checkpoint 的 ``latitudes.numpy`` / ``longitudes.numpy`` 提取

Reference
---------
- Lang et al., *AIFS — ECMWF's data-driven forecasting system*, arXiv:2406.01465
- anemoi-models: https://github.com/ecmwf/anemoi-models
- anemoi-graphs: https://github.com/ecmwf/anemoi-graphs
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from onescience.models.meta import ModelMetaData

LOG = logging.getLogger(__name__)

# ============================================================================
# 静态文件路径（相对于本文件所在目录）
# ============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_HERE, "aifs_config.json")

# grid-n320.npz 搜索路径优先级:
#   1. 模块目录（src/onescience/models/aifs/）—— 与模型代码同目录
#   2. 当前工作目录 —— download.sh 直接下载到项目根目录
#   3. 当前工作目录/weights/ —— 权重目录
_GRID_SEARCH_PATHS = [
    os.path.join(_HERE, "grid-n320.npz"),
    os.path.join(os.getcwd(), "./model/grid-n320.npz"),
    os.path.join(os.getcwd(), "weights", "grid-n320.npz"),
]


def _find_grid_file() -> str:
    """在多个候选位置中查找 N320 网格文件，返回第一个存在的路径。"""
    for p in _GRID_SEARCH_PATHS:
        if os.path.exists(p):
            return p
    return _GRID_SEARCH_PATHS[0]  # 默认返回模块目录路径（报错时给出明确提示）


# ============================================================================
# Metadata  (onescience convention)
# ============================================================================

@dataclass
class MetaData(ModelMetaData):
    name: str = "AIFS"
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    amp_gpu: bool = True
    bf16: bool = False
    onnx: bool = False              # flash_attn 不兼容 ONNX
    func_torch: bool = False
    auto_grad: bool = False
    var_dim: int = -1               # 每图节点的变量数


# ============================================================================
# Model
# ============================================================================

class AIFS(nn.Module):
    """AIFS v1.1 编码器-处理器-解码器 GNN。

    Parameters
    ----------
    checkpoint_path : str, optional
        ``pretrained=True`` 时必传——预训练 ``.ckpt`` 文件路径。
    device : str
        PyTorch 设备。默认 ``"cuda"``。
    pretrained : bool
        ``False`` (默认): 从零构建——读取本地 ``aifs_config.json`` +
        ``grid-n320.npz``, 不依赖 ``.ckpt`` 文件。
        ``True``: 从 ``.ckpt`` 加载预训练权重（向后兼容）。
    """

    metadata = MetaData()

    # ==================================================================
    # Factory: 真正从零（无需 .ckpt）
    # ==================================================================

    @classmethod
    def from_scratch(cls, device: str = "cuda") -> "AIFS":
        """从零创建 AIFS 模型——无需 ``.ckpt`` 文件。

        等价于 FengWu/Fuxi 的 ``model = Fengwu()`` 模式。
        所有权重随机初始化，架构配置和网格坐标从项目内静态文件加载。
        """
        return cls(device=device, pretrained=False)

    # ==================================================================
    # Construction
    # ==================================================================

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        pretrained: bool = False,
    ):
        super().__init__()

        if pretrained:
            if not checkpoint_path:
                raise ValueError(
                    "checkpoint_path is required when pretrained=True"
                )
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}"
                )
            self._init_from_pretrained(checkpoint_path, device)
        else:
            self._init_from_scratch(device)

    # ==================================================================
    # Mode 1: 从零构建（无需 .ckpt）
    # ==================================================================

    def _init_from_scratch(self, device: str):
        """用静态配置文件和标准网格坐标构建模型。"""
        LOG.info("Building AIFS from scratch (no checkpoint) …")

        # ---- 1. 加载静态架构配置 ----------------------------------------
        if not os.path.exists(_CONFIG_PATH):
            raise FileNotFoundError(
                f"Static config not found: {_CONFIG_PATH}. "
                "Run extract_static_config.py first."
            )
        grid_path = _find_grid_file()
        if not os.path.exists(grid_path):
            raise FileNotFoundError(
                f"Grid file not found.  Searched: {_GRID_SEARCH_PATHS}. "
                "Run 'bash download.sh' in the project root."
            )

        with open(_CONFIG_PATH) as f:
            config_data = json.load(f)
        LOG.info(
            "Loaded static config (%d KB)",
            os.path.getsize(_CONFIG_PATH) // 1024,
        )

        grid = np.load(grid_path)
        lat = np.asarray(grid["latitudes"], dtype=np.float32)
        lon = np.asarray(grid["longitudes"], dtype=np.float32)
        LOG.info(
            "Loaded N320 grid: %d nodes", len(lat),
        )

        # ---- 2. 构建 model_config + data_indices (官方类) ---------
        # IndexCollection 需要 OmegaConf, AnemoiModelEncProcDec 需要 DotDict.
        # 先从原始 JSON dict 创建 OmegaConf, 再转为 DotDict.
        raw_mc = config_data["model_config"]

        from anemoi.utils.config import DotDict

        # ---- 兼容性: 0.5.0 config → 0.9.0 API ---------------------------
        # 1. activation 参数已从 mapper/processor 移除（内部默认 GELU）
        for section in ("encoder", "decoder", "processor"):
            raw_mc["model"][section].pop("activation", None)
        raw_mc["model"].pop("activation", None)

        # 2. layer_kernels 从顶层移入各子模块 config
        lk = raw_mc["model"].get("layer_kernels", {})
        for section in ("encoder", "decoder", "processor"):
            if section in lk:
                raw_mc["model"][section]["layer_kernels"] = lk[section]

        model_config = DotDict(raw_mc)

        all_vars = config_data["dataset"]["variables"]
        name_to_index = {name: i for i, name in enumerate(all_vars)}

        from anemoi.models.data_indices.collection import IndexCollection
        from omegaconf import OmegaConf
        data_indices = IndexCollection(
            config=OmegaConf.create(raw_mc),
            name_to_index=name_to_index,
        )
        graph_data = self._build_graph(lat, lon)

        # 提取 area_weight 作为 node_weights
        area_wt = (
            graph_data["data"]["area_weight"].cpu().numpy().squeeze()
        )

        # ---- 4. 实例化 AnemoiModelEncProcDec（随机权重）---------------
        # 训练时直接用裸模型——AnemoiModelInterface 自带 normalizer
        # 预处理，会导致训练时双重归一化。保存 checkpoint 时再包装。
        LOG.info("Instantiating AnemoiModelEncProcDec (random weights) …")
        from anemoi.models.models.encoder_processor_decoder import \
            AnemoiModelEncProcDec

        self._model = AnemoiModelEncProcDec(
            model_config=model_config,
            data_indices=data_indices,
            statistics={},
            graph_data=graph_data,
            truncation_data={},
        ).to(device)

        # 保存接口构建所需素材（_save 中构建 AnemoiModelInterface 用）
        self._interface_config = model_config
        self._interface_graph_data = graph_data
        self._interface_data_indices = data_indices

        # ---- 5. 提取变量排序和坐标 -----------------------------------
        self._meta = config_data
        self._arrays = {
            "latitudes": lat,
            "longitudes": lon,
            "area_weight": area_wt,
        }
        self._extract_metadata()

        LOG.info(
            "AIFS (from scratch) ready: %d → %d vars, %d grid pts, "
            "%.1f M params",
            len(self._input_vars),
            len(self._output_vars),
            len(lat),
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def _build_graph(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> "HeteroData":
        """构建 AIFS 的 HeteroData 图（N320 ↔ o96）。

        完全使用 ``anemoi-graphs`` 官方 API, 不依赖任何外部数据文件。
        节点坐标来自标准 ECMWF 网格定义, 边由纯几何算法计算。
        """
        from torch_geometric.data import HeteroData

        from anemoi.graphs.edges.builders.cutoff import CutOffEdges
        from anemoi.graphs.edges.builders.knn import KNNEdges
        from anemoi.graphs.nodes.builders.from_reduced_gaussian import \
            ReducedGaussianGridNodes
        from anemoi.graphs.nodes.builders.from_vectors import LatLonNodes

        graph = HeteroData()

        # -- 数据节点 (N320: 542,080 nodes) --
        data_builder = LatLonNodes(
            latitudes=lat, longitudes=lon, name="data",
        )
        data_attrs = {
            "area_weight": {
                "_target_": "anemoi.graphs.nodes.attributes.SphericalAreaWeights",
                "fill_value": 0,
                "norm": "unit-max",
            },
        }
        graph = data_builder.update_graph(graph, attrs_config=data_attrs)

        # -- 隐藏节点 (o96: ~8,000 nodes) --
        hidden_builder = ReducedGaussianGridNodes(
            grid="o96", name="hidden",
        )
        graph = hidden_builder.update_graph(graph, attrs_config={})

        # -- data → hidden 边 (CutOffEdges, cutoff_factor=0.6) --
        d2h_attrs = {
            "edge_dirs": {
                "_target_": "anemoi.graphs.edges.attributes.EdgeDirection",
                "norm": "unit-std",
            },
            "edge_length": {
                "_target_": "anemoi.graphs.edges.attributes.EdgeLength",
                "norm": "unit-std",
            },
        }
        d2h_builder = CutOffEdges(
            source_name="data",
            target_name="hidden",
            cutoff_factor=0.6,
        )
        graph = d2h_builder.update_graph(graph, attrs_config=None)
        graph = d2h_builder.register_attributes(graph, d2h_attrs)

        # -- hidden → data 边 (KNNEdges, K=3) --
        h2d_attrs = {
            "edge_dirs": {
                "_target_": "anemoi.graphs.edges.attributes.EdgeDirection",
                "norm": "unit-std",
            },
            "edge_length": {
                "_target_": "anemoi.graphs.edges.attributes.EdgeLength",
                "norm": "unit-std",
            },
        }
        h2d_builder = KNNEdges(
            source_name="hidden",
            target_name="data",
            num_nearest_neighbours=3,
        )
        graph = h2d_builder.update_graph(graph, attrs_config=None)
        graph = h2d_builder.register_attributes(graph, h2d_attrs)

        LOG.info(
            "Graph built: data(%d nodes) ↔ hidden(%d nodes), "
            "d→h edges=%d, h→d edges=%d",
            graph["data"].num_nodes,
            graph["hidden"].num_nodes,
            graph["data", "to", "hidden"].edge_index.shape[1],
            graph["hidden", "to", "data"].edge_index.shape[1],
        )
        return graph

    # ==================================================================
    # Mode 2: 加载预训练 (pretrained=True, 向后兼容)
    # ==================================================================

    def _init_from_pretrained(self, checkpoint_path: str, device: str):
        """从 checkpoint 加载完整序列化模型（图 + 权重 + 索引）。"""
        LOG.info("Loading AIFS checkpoint from %s …", checkpoint_path)
        _cp = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
        self._model = _cp.to(device)

        # 提取元数据
        self._meta, self._arrays = self._read_checkpoint_metadata(
            checkpoint_path,
        )
        self._extract_metadata()

        LOG.info(
            "AIFS (pretrained) ready: %d → %d vars, %d grid pts, "
            "%.1f M params",
            len(self._input_vars),
            len(self._output_vars),
            len(self.latitudes),
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    # ==================================================================
    # Metadata helpers
    # ==================================================================

    @staticmethod
    def _read_checkpoint_metadata(checkpoint_path: str):
        """从 checkpoint ZIP 中读取 JSON 元数据和支持数组。"""
        import zipfile
        from anemoi.utils.checkpoints import load_supporting_arrays

        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            # 支持新旧两种元数据文件名
            metadata = None
            for name in ["anemoi.json", "ai-models.json"]:
                for fname in zf.namelist():
                    if fname.endswith(name):
                        metadata = json.load(zf.open(fname, "r"))
                        break
                if metadata:
                    break

            if metadata is None:
                raise FileNotFoundError(
                    f"No metadata JSON found in {checkpoint_path}"
                )

            arrays = load_supporting_arrays(
                zf, metadata.get("supporting_arrays_paths", {}),
            )
            return metadata, arrays

    def _extract_metadata(self):
        """从已加载的元数据中提取变量排序和网格坐标。"""
        all_vars: List[str] = self._meta["dataset"]["variables"]
        di = self._meta["data_indices"]["data"]
        self._input_vars = [all_vars[i] for i in di["input"]["full"]]
        self._output_vars = [all_vars[i] for i in di["output"]["full"]]
        self._aifs_name_to_ds_idx: Dict[str, int] = {
            n: i for i, n in enumerate(all_vars)
        }

        lat = np.asarray(self._arrays["latitudes"], dtype=np.float32)
        lon = np.asarray(self._arrays["longitudes"], dtype=np.float32)
        self.register_buffer("latitudes", torch.from_numpy(lat))
        self.register_buffer("longitudes", torch.from_numpy(lon))

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def num_input_vars(self) -> int:
        """输入张量中的变量数 (103)。"""
        return len(self._input_vars)

    @property
    def num_output_vars(self) -> int:
        """输出张量中的变量数 (102)。"""
        return len(self._output_vars)

    @property
    def num_grid_points(self) -> int:
        """N320 网格节点数 (542,080)。"""
        return len(self.latitudes)

    @property
    def node_weights(self) -> Optional[np.ndarray]:
        """损失函数中每节点面积权重，无则为 None。"""
        for k in ["node_weights", "area_weight"]:
            if k in self._arrays:
                return np.asarray(
                    self._arrays[k], dtype=np.float32,
                ).squeeze()
        return None

    @property
    def input_variables(self) -> List[str]:
        """输入张量中 AIFS 变量名的有序列表。"""
        return self._input_vars

    @property
    def output_variables(self) -> List[str]:
        """输出张量中 AIFS 变量名的有序列表。"""
        return self._output_vars

    def aifs_name_to_dataset_index(self, name: str) -> int:
        """返回 *name* 在 115 变量数据集中的位置。"""
        return self._aifs_name_to_ds_idx.get(name, -1)

    def input_name_to_channel(self, name: str) -> int:
        """返回 *name* 在输入张量中的通道索引。"""
        try:
            return self._input_vars.index(name)
        except ValueError:
            return -1

    def output_name_to_channel(self, name: str) -> int:
        """返回 *name* 在输出张量中的通道索引。"""
        try:
            return self._output_vars.index(name)
        except ValueError:
            return -1

    # ------------------------------------------------------------------
    # Forward  (匹配官方 AnemoiModelEncProcDec.forward)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        等价于 ``AnemoiModelEncProcDec.forward(x, model_comm_group=None)``
        在单 GPU 训练/推理时的行为。

        Parameters
        ----------
        x : torch.Tensor
            归一化输入。
            - 4-D: ``(B, T, G, V_in)`` — 自动提升为 5-D。
            - 5-D: ``(B, T, E, G, V_in)`` — ensemble 维度保留。

        Returns
        -------
        torch.Tensor
            - 4-D 输入 → ``(B, G, V_out)``
            - 5-D 输入 → ``(B, E, G, V_out)``
        """
        if x.ndim == 4:
            x = x.unsqueeze(2)          # (B, T, G, V) → (B, T, 1, G, V)

        out = self._model(x)            # (B, E, G, V_out)

        if out.shape[1] == 1:
            out = out.squeeze(1)        # (B, 1, G, V) → (B, G, V)
        return out

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推理前向传播（无梯度，eval 模式，autocast fp16）。

        FlashAttention 要求 fp16/bf16 输入，用 autocast 自动转换。
        """
        was_training = self.training
        self.eval()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = self.forward(x)
        if was_training:
            self.train()
        return out.float()

    # ------------------------------------------------------------------
    # Train / eval propagation
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """将 train/eval 模式传播到被封装的官方模型。"""
        super().train(mode)
        self._model.train(mode)
        return self

    def eval(self):
        return self.train(False)
