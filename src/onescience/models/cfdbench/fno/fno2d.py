from typing import List, Dict, Optional

import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .Adam import Adam
from .utilities3 import MatReader, count_params
from ..base_model import AutoCfdModel

torch.manual_seed(0)
np.random.seed(0)


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(  # type: ignore
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(  # type: ignore
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y)
        # -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FnoBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        modes1: int,
        modes2: int,
        act_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.modes1 = modes1
        self.modes2 = modes2
        self.act_fn = act_fn

        self.conv0 = SpectralConv2d_fast(
            self.in_chan, self.out_chan, self.modes1, self.modes2
        )
        self.w0 = nn.Conv2d(self.in_chan, self.out_chan, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x


class Fno2d(AutoCfdModel):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        n_case_params: int,
        loss_fn: nn.Module,
        num_layers: int,
        modes1: int = 12,
        modes2: int = 12,
        hidden_dim: int = 20,
        padding: Optional[int] = None,
    ):
        super().__init__(loss_fn)

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1
            and self.fc2 .
        """
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.num_layers = num_layers
        self.modes1 = modes1
        self.modes2 = modes2
        self.hidden_dim = hidden_dim
        self.padding = padding  # pad the domain if input is non-periodic

        self.act_fn = nn.GELU()
        # Channel projection into `hidden_dim` channels
        # +1 for mask, +2 for coordinates
        self.fc0 = nn.Conv2d(
            in_chan + 1 + 2 + n_case_params,
            self.hidden_dim,
            1,
            1,
            0,
        )
        # input channel is 12: the solution of the previous 10
        # timesteps + 2 locations
        # (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        # FNO blocks
        blocks = []
        for _ in range(self.num_layers):
            blocks.append(
                FnoBlock(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.modes1,
                    self.modes2,
                    self.act_fn,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.fc1 = nn.Conv2d(self.hidden_dim, 128, 1, 1, 0)
        self.fc2 = nn.Conv2d(128, self.out_chan, 1, 1, 0)

    def forward(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
        label: Optional[Tensor] = None,
    ) -> Dict:
        """
        Args:
        - input: (b, c, h, w)
        - labels: (b, c, h, w)
        - mask: (b, h, w), a binary mask for indicating the geometry
                1 for interior, 0 for obstacles.

        Returns: a tensor of shape (b, c, h, w), the solution at
            the next timestep
        """
        batch_size, n_chan, height, width = inputs.shape

        if mask is None:
            # When there is no mask, we assume that there is no obstacles.
            mask = torch.ones((batch_size, 1, height, width)).to(inputs.device)
        else:
            if mask.dim() == 3:  # (B, h, w)
                mask = mask.unsqueeze(1)  # (B, 1, h, w)
        inputs = torch.cat([inputs, mask], dim=1)  # (B, c + 1, h, w)

        # Physical properties
        props = case_params  # (B, p)
        props = props.unsqueeze(-1).unsqueeze(-1)  # (B, p, 1, 1)
        props = props.repeat(1, 1, height, width)  # (B, p, H, W)

        # Append (x, y) coordinates to every location
        grid = self.get_coords(inputs.shape, inputs.device)  # (b, 2, h, w)
        inputs = torch.cat(
            (inputs, grid, props), dim=1
        )  # (b, c + 2 + 2, h, w)

        # Project channels
        inputs = self.fc0(inputs)  # (b, hidden_dim, h, w)
        # x = x.permute(0, 3, 1, 2)  # (b, c, h, w)?
        if self.padding is not None:
            # pad the domain if input is non-periodic
            inputs = F.pad(inputs, [0, self.padding, 0, self.padding])

        inputs = self.blocks(inputs)  # (b, hidden_dim, h, w)
        if self.padding is not None:
            # pad the domain if inputis non-periodic
            inputs = inputs[..., : -self.padding, : -self.padding]

        inputs = self.fc1(inputs)  # (b, 128, h, w)
        inputs = self.act_fn(inputs)
        preds = self.fc2(inputs)  # (b, c_out, h, w)

        # Masked locations are not prediction
        preds = preds * mask

        if label is not None:
            label = label * mask
            loss = self.loss_fn(preds=preds, labels=label)
            return dict(
                preds=preds,
                loss=loss,
            )
        return dict(preds=preds)

    def get_coords(self, shape, device):
        """
        Return a tensor of shape (b, 2, h, w) such that the element at
        [:, :, i, j] is the (x, y) coordinates at the grid location (i, j).
        """
        bsz, c, size_x, size_y = shape
        grid_x = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        grid_x = grid_x.reshape(1, 1, size_x, 1).repeat([bsz, 1, 1, size_y])
        grid_y = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        grid_y = grid_y.reshape(1, 1, 1, size_y).repeat([bsz, 1, size_x, 1])
        coords = torch.cat([grid_x, grid_y], dim=1).to(device)  # (b, 2, h, w)
        return coords

    def generate(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        outputs = self.forward(
            inputs=inputs, case_params=case_params, mask=mask
        )  # (b, c, h, w)
        preds = outputs["preds"]
        return preds

    def generate_many(
        self, inputs: Tensor, case_params: Tensor, mask: Tensor, steps: int
    ) -> List[Tensor]:
        """
        Args:
            x (Tensor): (c, h, w)
            case_params (Tensor): (p)
            mask (Tensor): (h, w), 1 for interior, 0 for obstacles.
        Returns:
            output: (steps, c, h, w)
        """
        assert len(inputs.shape) == len(case_params.shape) + 2
        if inputs.dim() == 3:
            # Add a dimension for batch size of 1
            inputs = inputs.unsqueeze(0)
            case_params = case_params.unsqueeze(0)
            mask = mask.unsqueeze(0)
        assert inputs.shape[0] == case_params.shape[0] == mask.shape[0]

        cur_frame = inputs  # (b, c, h, w)
        preds = []
        for _ in range(steps):
            cur_frame = self.generate(
                inputs=cur_frame, case_params=case_params, mask=mask
            )
            preds.append(cur_frame)
        return preds


if __name__ == "__main__":
    TRAIN_PATH = "data/ns_data_V100_N1000_T50_1.mat"
    TEST_PATH = "data/ns_data_V100_N1000_T50_2.mat"

    ntrain = 1000
    ntest = 200

    modes = 12
    width = 20

    batch_size = 20
    batch_size2 = batch_size

    epochs = 500
    learning_rate = 0.001
    scheduler_step = 100
    scheduler_gamma = 0.5

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    path = (
        "ns_fourier_2d_rnn_V10000_T20_N"
        + str(ntrain)
        + "_ep"
        + str(epochs)
        + "_m"
        + str(modes)
        + "_w"
        + str(width)
    )
    path_model = "model/" + path
    path_train_err = "results/" + path + "train.txt"
    path_test_err = "results/" + path + "test.txt"
    path_image = "image/" + path

    sub = 1
    S = 64
    T_in = 10
    T = 10
    step = 1

    ################################################################
    # load data
    ################################################################

    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field("u")[:ntrain, ::sub, ::sub, :T_in]  # type: ignore  # noqa
    train_u = reader.read_field("u")[:ntrain, ::sub, ::sub, T_in : T + T_in]  # type: ignore # noqa

    reader = MatReader(TEST_PATH)
    test_a = reader.read_field("u")[-ntest:, ::sub, ::sub, :T_in]  # type: ignore  # noqa
    test_u = reader.read_field("u")[-ntest:, ::sub, ::sub, T_in : T + T_in]  # type: ignore  # noqa

    print(train_u.shape)  # type: ignore
    print(test_u.shape)  # type: ignore
    assert S == train_u.shape[-2]  # type: ignore
    assert T == train_u.shape[-1]  # type: ignore

    train_a = train_a.reshape(ntrain, S, S, T_in)  # type: ignore
    test_a = test_a.reshape(ntest, S, S, T_in)  # type: ignore

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u),
        batch_size=batch_size,
        shuffle=False,
    )

    ################################################################
    # training and evaluation
    ################################################################

    model = Fno2d(modes, modes, width).cuda()  # type: ignore
    # model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

    print(count_params(model))
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # type: ignore  # noqa
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma  # type: ignore  # noqa
    )
