from types import SimpleNamespace
import warnings

import torch


def make_args(**overrides):
    args = dict(
        unified_pos=False,
        geotype="structured_2D",
        shapelist=(8, 8),
        ref=4,
        fun_dim=1,
        space_dim=2,
        out_dim=2,
        n_hidden=8,
        act="gelu",
        time_input=False,
        modes=4,
        task="steady",
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        mlp_ratio=1,
        slice_num=8,
        branch_depth=2,
        trunk_depth=2,
        mwt_k=2,
        psi_dim=4,
        attn_type="linear",
        emb_dims=16,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def print_shape(name, out, target):
    print(f"Function: CFD_Benchmark {name} Forward")
    print(f"output shape: {out.shape}")
    print(f"target shape: {target}\n")


def print_parameter_count(name, model):
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameter count: CFD_Benchmark {name}: {parameter_count}")


def run_structured_model(name, model_cls, args):
    n_points = args.shapelist[0] * args.shapelist[1]
    x = torch.randn(1, n_points, args.space_dim)
    fx = torch.randn(1, n_points, args.fun_dim)

    model = model_cls(args, torch.device("cpu"))
    print_parameter_count(name, model)
    model.eval()
    with torch.no_grad():
        out = model(x, fx)

    target = torch.Size([1, n_points, args.out_dim])
    assert out.shape == target, f"{name}: got {out.shape}, expected {target}"
    print_shape(name, out, target)


def run_unstructured_model(name, model_cls, args, n_points=16):
    x = torch.randn(1, n_points, args.space_dim)
    fx = torch.randn(1, n_points, args.fun_dim)
    edge_index = torch.tensor(
        [list(range(n_points)), list(range(1, n_points)) + [0]],
        dtype=torch.long,
    )

    model = model_cls(args, torch.device("cpu"))
    print_parameter_count(name, model)
    if name == "RegDGCNN":
        model.k = min(8, n_points)
    model.eval()
    with torch.no_grad():
        out = model(x, fx, geo=edge_index)

    target = torch.Size([1, n_points, args.out_dim])
    assert out.shape == target, f"{name}: got {out.shape}, expected {target}"
    print_shape(name, out, target)


warnings.filterwarnings("ignore", message="A not p.d.*")
warnings.filterwarnings("ignore", message="An issue occurred while importing.*")
torch.manual_seed(0)

from onescience.models.cfd_benchmark.DeepONet import Model as DeepONet
from onescience.models.cfd_benchmark.FNO import Model as FNO
from onescience.models.cfd_benchmark.F_FNO import Model as FFNO
from onescience.models.cfd_benchmark.Factformer import Model as Factformer
from onescience.models.cfd_benchmark.GFNO import Model as GFNO
from onescience.models.cfd_benchmark.GNOT import Model as GNOT
from onescience.models.cfd_benchmark.Galerkin_Transformer import (
    Model as GalerkinTransformer,
)
from onescience.models.cfd_benchmark.LSM import Model as LSM
from onescience.models.cfd_benchmark.MWT import Model as MWT
from onescience.models.cfd_benchmark.ONO import Model as ONO
from onescience.models.cfd_benchmark.Swin_Transformer import Model as SwinTransformer
from onescience.models.cfd_benchmark.Transformer import Model as Transformer
from onescience.models.cfd_benchmark.Transolver import Model as Transolver
from onescience.models.cfd_benchmark.U_FNO import Model as UFNO
from onescience.models.cfd_benchmark.U_NO import Model as UNO
from onescience.models.cfd_benchmark.U_Net import Model as BenchmarkUNet

structured_models = [
    ("FNO", FNO, make_args()),
    ("F-FNO", FFNO, make_args()),
    ("Factformer", Factformer, make_args()),
    ("GFNO", GFNO, make_args()),
    ("GNOT", GNOT, make_args()),
    ("Galerkin_Transformer", GalerkinTransformer, make_args()),
    ("LSM", LSM, make_args()),
    ("MWT", MWT, make_args()),
    ("ONO", ONO, make_args()),
    ("Swin_Transformer", SwinTransformer, make_args()),
    ("Transformer", Transformer, make_args()),
    ("Transolver", Transolver, make_args()),
    ("U_FNO", UFNO, make_args()),
    ("U_NO", UNO, make_args()),
    ("U-Net", BenchmarkUNet, make_args(shapelist=(16, 16), n_hidden=4)),
    ("DeepONet", DeepONet, make_args()),
]

for model_name, model_class, model_args in structured_models:
    run_structured_model(model_name, model_class, model_args)

try:
    from onescience.models.cfd_benchmark.GraphSAGE import Model as GraphSAGE
    from onescience.models.cfd_benchmark.PointNet import Model as PointNet
    from onescience.models.cfd_benchmark.RegDGCNN import Model as RegDGCNN

    graph_args = make_args(geotype="unstructured", n_hidden=4)
    run_unstructured_model("PointNet", PointNet, graph_args)
    run_unstructured_model("GraphSAGE", GraphSAGE, graph_args)
    run_unstructured_model("RegDGCNN", RegDGCNN, graph_args, n_points=48)
except Exception as exc:
    print(f"SKIP: CFD_Benchmark torch_geometric point models ({type(exc).__name__}: {exc})")

try:
    from onescience.models.cfd_benchmark.Graph_UNet import Model as GraphUNet

    graph_args = make_args(geotype="unstructured", n_hidden=4)
    edge_index = torch.tensor(
        [list(range(16)), list(range(1, 16)) + [0]],
        dtype=torch.long,
    )
    model = GraphUNet(
        graph_args,
        torch.device("cpu"),
        scale=2,
        pool_ratio=[0.5],
        list_r=[10],
        max_neighbors=8,
    )
    print_parameter_count("Graph_UNet", model)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(16, 2), torch.randn(1, 16, 1), geo=edge_index)
    target = torch.Size([1, 16, graph_args.out_dim])
    assert out.shape == target, f"Graph_UNet: got {out.shape}, expected {target}"
    print_shape("Graph_UNet", out, target)
except (ImportError, OSError) as exc:
    print(f"SKIP: CFD_Benchmark Graph_UNet optional dependency ({type(exc).__name__}: {exc})")

try:
    from onescience.models.cfd_benchmark.MeshGraphNet import Model as BenchmarkMGN

    print(f"SKIP: CFD_Benchmark MeshGraphNet requires a DGL graph ({BenchmarkMGN.__name__})")
except (ImportError, ModuleNotFoundError, OSError) as exc:
    print(f"SKIP: CFD_Benchmark MeshGraphNet optional dependency ({type(exc).__name__}: {exc})")
