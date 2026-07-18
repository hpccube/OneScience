import torch

from onescience.models.factformer import FactFormer2D
from onescience.models.gfno import GFNO
from onescience.models.kno import KNO2DNavierStokes
from onescience.models.ono import ONO
from onescience.models.uno import UNO
from onescience.modules.equivariant.group_conv import GroupEquivariantConv2d
from onescience.modules.fourier.fno_layers import SpectralConv2d
from onescience.modules.fourier.group_spectral import GSpectralConv2d
from onescience.modules.koopman import (
    DecoderConv2D,
    EncoderConv2D,
    KoopmanOperator2D,
)
from onescience.modules.mlp.GMLP import GroupEquivariantMLP2d
from onescience.modules.mlp.MLP import StandardMLP
from onescience.modules.transformer.factformer_block import Factformer_block
from onescience.modules.transformer.orthogonal_neural_block import OrthogonalNeuralBlock


def structured_grid(batch_size: int, height: int, width: int) -> torch.Tensor:
    x_axis = torch.linspace(0.0, 1.0, height)
    y_axis = torch.linspace(0.0, 1.0, width)
    grid_x, grid_y = torch.meshgrid(x_axis, y_axis, indexing="ij")
    points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    return points.unsqueeze(0).repeat(batch_size, 1, 1)


def test_gfno_forward_shape():
    model = GFNO(
        in_dim=3,
        out_dim=1,
        spatial_shape=(16, 16),
        hidden_dim=4,
        modes=2,
        num_layers=1,
    )
    assert isinstance(model.preprocess, StandardMLP)
    assert isinstance(model.p, GroupEquivariantConv2d)
    assert isinstance(model.spectral_layers[0], GSpectralConv2d)
    assert isinstance(model.mlp_layers[0], GroupEquivariantMLP2d)
    output = model(structured_grid(1, 16, 16), torch.randn(1, 256, 3))
    assert output.shape == (1, 256, 1)
    assert torch.isfinite(output).all()


def test_ono_forward_shape():
    model = ONO(
        in_dim=3,
        out_dim=1,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        attn_type="linear",
        psi_dim=2,
    )
    assert isinstance(model.preprocess_x, StandardMLP)
    assert isinstance(model.preprocess_z, StandardMLP)
    assert isinstance(model.blocks[0], OrthogonalNeuralBlock)
    output = model(structured_grid(2, 4, 4), torch.randn(2, 16, 3))
    assert output.shape == (2, 16, 1)
    assert torch.isfinite(output).all()


def test_uno_forward_shape():
    model = UNO(
        in_dim=3,
        out_dim=1,
        spatial_shape=(32, 32),
        hidden_dim=2,
        modes=2,
    )
    assert isinstance(model.preprocess, StandardMLP)
    assert isinstance(model.process1_down, SpectralConv2d)
    output = model(structured_grid(1, 32, 32), torch.randn(1, 1024, 3))
    assert output.shape == (1, 1024, 1)
    assert torch.isfinite(output).all()


def test_factformer_forward_shape():
    model = FactFormer2D(
        in_dim=3,
        out_dim=1,
        spatial_shape=(8, 8),
        hidden_dim=8,
        depth=1,
        heads=2,
        max_latent_steps=2,
    )
    assert isinstance(model.preprocess, StandardMLP)
    assert isinstance(model.blocks[0], Factformer_block)
    assert isinstance(model.propagator, StandardMLP)
    assert isinstance(model.to_out, StandardMLP)
    output = model(
        structured_grid(1, 8, 8),
        torch.randn(1, 64, 3),
        latent_steps=2,
    )
    assert output.shape == (1, 64, 2)
    assert torch.isfinite(output).all()


def test_kno_uses_public_koopman_modules():
    encoder = EncoderConv2D(t_len=3, op_size=4)
    decoder = DecoderConv2D(t_len=1, op_size=4)
    operator = KoopmanOperator2D(op_size=4, modes_x=2, modes_y=2)
    assert encoder(torch.randn(1, 8, 8, 3)).shape == (1, 8, 8, 4)
    assert decoder(torch.randn(1, 8, 8, 4)).shape == (1, 8, 8, 1)
    assert operator(torch.randn(1, 4, 8, 8)).shape == (1, 4, 8, 8)

    model = KNO2DNavierStokes(
        input_channels=3,
        output_channels=1,
        spatial_shape=(8, 8),
        op_size=4,
        modes_x=2,
        modes_y=2,
        decompose=1,
    )
    output = model(structured_grid(1, 8, 8), torch.randn(1, 64, 3))
    assert output.shape == (1, 64, 1)
    assert torch.isfinite(output).all()
