import torch

from onescience.models.pdenneval.deeponet import (
    DeepONet,
    DeepONetCartesianProd,
    DeepONetCartesianProd1D,
    DeepONetCartesianProd2D,
    MLP,
    Modified_MLP,
)
from onescience.models.pdenneval.fno import (
    FNO1d as EvalFNO1d,
    FNO2d as EvalFNO2d,
    FNO3d as EvalFNO3d,
    FNO_maxwell as EvalFNO_maxwell,
)
from onescience.models.pdenneval.pino_fno import (
    FNO1d as PINOFNO1d,
    FNO2d as PINOFNO2d,
    FNO3d as PINOFNO3d,
)
from onescience.models.pdenneval.unet import UNet1d, UNet2d, UNet3d
from onescience.models.pdenneval.uno import UNO1d, UNO2d, UNO3d, UNO_maxwell


def print_shape(name, out, target):
    print(f"Function: {name} Forward")
    print(f"output shape: {out.shape}")
    print(f"target shape: {target}\n")


def print_num_params(name, model):
    num_params = sum(param.numel() for param in model.parameters())
    print(f"{name} parameters: {num_params}")


def run_forward(name, model, args, target):
    print_num_params(name, model)
    model.eval()
    with torch.no_grad():
        out = model(*args)
    assert out.shape == target, f"{name}: got {out.shape}, expected {target}"
    print_shape(name, out, target)


torch.manual_seed(0)

deeponet_models = [
    (
        "PDENNEval MLP",
        MLP([3, 8, 2], "relu", "Glorot normal"),
        (torch.randn(4, 3),),
        torch.Size([4, 2]),
    ),
    (
        "PDENNEval Modified_MLP",
        Modified_MLP([3, 8, 2], "relu", "Glorot normal"),
        (torch.randn(4, 3),),
        torch.Size([4, 2]),
    ),
    (
        "PDENNEval DeepONet",
        DeepONet([3, 8, 4], [2, 8, 4], "relu", "Glorot normal"),
        ((torch.randn(5, 3), torch.randn(5, 2)),),
        torch.Size([5, 1]),
    ),
    (
        "PDENNEval DeepONetCartesianProd",
        DeepONetCartesianProd([3, 8, 4], [2, 8, 4], "relu", "Glorot normal"),
        ((torch.randn(3, 3), torch.randn(6, 2)),),
        torch.Size([3, 6]),
    ),
    (
        "PDENNEval DeepONetCartesianProd1D",
        DeepONetCartesianProd1D(8, 2, 1, 2, activation="relu"),
        ((torch.randn(1, 8, 2), torch.randn(8, 1)),),
        torch.Size([1, 8, 2]),
    ),
    (
        "PDENNEval DeepONetCartesianProd2D",
        DeepONetCartesianProd2D(6, 2, 2, 2, activation="relu"),
        ((torch.randn(1, 6, 6, 2), torch.randn(6, 6, 2)),),
        torch.Size([1, 6, 6, 2]),
    ),
]

for name, model, args, target in deeponet_models:
    run_forward(name, model, args, target)


eval_fno_models = [
    (
        "PDENNEval FNO1d",
        EvalFNO1d(1, modes=4, width=8, initial_step=2),
        (torch.randn(1, 16, 2), torch.randn(1, 16, 1)),
        torch.Size([1, 16, 1, 1]),
    ),
    (
        "PDENNEval FNO2d",
        EvalFNO2d(1, modes1=4, modes2=4, width=8, initial_step=2),
        (torch.randn(1, 12, 12, 2), torch.randn(1, 12, 12, 2)),
        torch.Size([1, 12, 12, 1, 1]),
    ),
    (
        "PDENNEval FNO3d",
        EvalFNO3d(1, modes1=2, modes2=2, modes3=2, width=6, initial_step=2),
        (torch.randn(1, 8, 8, 8, 2), torch.randn(1, 8, 8, 8, 3)),
        torch.Size([1, 8, 8, 8, 1, 1]),
    ),
    (
        "PDENNEval FNO_maxwell",
        EvalFNO_maxwell(1, modes1=2, modes2=2, modes3=2, width=6, initial_step=2),
        (torch.randn(1, 8, 8, 8, 2), torch.randn(1, 8, 8, 8, 3)),
        torch.Size([1, 8, 8, 8, 1, 1]),
    ),
]

for name, model, args, target in eval_fno_models:
    run_forward(name, model, args, target)


pino_fno_models = [
    (
        "PDENNEval PINO FNO1d",
        PINOFNO1d([4, 4, 4], width=8, fc_dim=16, in_dim=3, out_dim=2),
        (torch.randn(1, 16, 3),),
        torch.Size([1, 16, 2]),
    ),
    (
        "PDENNEval PINO FNO2d",
        PINOFNO2d([4, 4, 4], [4, 4, 4], width=8, fc_dim=16, in_dim=4, out_dim=2),
        (torch.randn(1, 12, 12, 4),),
        torch.Size([1, 12, 12, 2]),
    ),
    (
        "PDENNEval PINO FNO3d",
        PINOFNO3d(
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            width=6,
            fc_dim=16,
            in_dim=5,
            out_dim=2,
        ),
        (torch.randn(1, 8, 8, 8, 5),),
        torch.Size([1, 8, 8, 8, 2]),
    ),
]

for name, model, args, target in pino_fno_models:
    run_forward(name, model, args, target)


unet_models = [
    (
        "PDENNEval UNet1d",
        UNet1d(in_channels=3, out_channels=2, init_features=2),
        (torch.randn(1, 3, 32),),
        torch.Size([1, 2, 32]),
    ),
    (
        "PDENNEval UNet2d",
        UNet2d(in_channels=3, out_channels=2, init_features=2),
        (torch.randn(1, 3, 32, 32),),
        torch.Size([1, 2, 32, 32]),
    ),
    (
        "PDENNEval UNet3d",
        UNet3d(in_channels=2, out_channels=1, init_features=1),
        (torch.randn(1, 2, 16, 16, 16),),
        torch.Size([1, 1, 16, 16, 16]),
    ),
]

for name, model, args, target in unet_models:
    run_forward(name, model, args, target)


uno_models = [
    (
        "PDENNEval UNO1d",
        UNO1d(num_channels=1, width=4, pad=9, factor=1, initial_step=2),
        (torch.randn(1, 48, 2), torch.randn(1, 48, 1)),
        torch.Size([1, 48, 1, 1]),
    ),
    (
        "PDENNEval UNO2d",
        UNO2d(num_channels=1, width=4, pad=6, factor=1, initial_step=2),
        (torch.randn(1, 58, 58, 2), torch.randn(1, 58, 58, 2)),
        torch.Size([1, 58, 58, 1, 1]),
    ),
    (
        "PDENNEval UNO3d",
        UNO3d(num_channels=1, width=2, pad=5, factor=1, initial_step=2),
        (torch.randn(1, 59, 59, 59, 2), torch.randn(1, 59, 59, 59, 3)),
        torch.Size([1, 59, 59, 59, 1, 1]),
    ),
    (
        "PDENNEval UNO_maxwell",
        UNO_maxwell(num_channels=1, width=2, pad=5, factor=1, initial_step=2),
        (torch.randn(1, 59, 59, 59, 2), torch.randn(1, 59, 59, 59, 3)),
        torch.Size([1, 59, 59, 59, 1, 1]),
    ),
]

for name, model, args, target in uno_models:
    run_forward(name, model, args, target)


try:
    from torch_geometric.data import Data

    from onescience.models.pdenneval.mpnn import MPNN
    from onescience.utils.pdenneval.mpnn_utils import PDE

    pde = PDE(
        "unit_test_pde",
        variables={"nu": 0.1},
        temporal_domain=(0.0, 1.0),
        resolution_t=10,
        spatial_domain=[(0.0, 1.0)],
        resolution=[4],
    )
    mpnn = MPNN(
        pde=pde,
        time_window=10,
        hidden_features=128,
        hidden_layers=1,
        eq_variables={"nu": 0.1},
    )
    graph = Data(
        x=torch.randn(4, 10, 1),
        x_pos=torch.linspace(0.0, 1.0, 4).unsqueeze(-1),
        t_pos=torch.zeros(4),
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 0, 3, 0, 1, 2]],
            dtype=torch.long,
        ),
        batch=torch.zeros(4, dtype=torch.long),
        variables=torch.full((4, 1), 0.1),
    )
    run_forward(
        "PDENNEval MPNN",
        mpnn,
        (graph, 0),
        torch.Size([4, 10]),
    )
except (ImportError, ModuleNotFoundError, OSError) as exc:
    print(f"SKIP: PDENNEval MPNN optional dependency ({type(exc).__name__}: {exc})")
