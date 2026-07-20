# -*- coding: utf-8 -*-
"""
Ref:
    - https://github.com/mir-group/nequip
    - https://www.nature.com/articles/s41467-022-29939-5
"""

import math
from typing import Optional

import torch
from e3nn.math import soft_one_hot_linspace
from torch import nn

from onescience.modules.func_utils.mattersim_jit import compile_mode


class e3nn_basias(nn.Module):
    def __init__(
        self,
        r_max: float,
        r_min: Optional[float] = None,
        e3nn_basis_name: str = "gaussian",
        num_basis: int = 8,
    ):
        super().__init__()
        self.r_max = r_max
        self.r_min = r_min if r_min is not None else 0.0
        self.e3nn_basis_name = e3nn_basis_name
        self.num_basis = num_basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return soft_one_hot_linspace(
            x,
            start=self.r_min,
            end=self.r_max,
            number=self.num_basis,
            basis=self.e3nn_basis_name,
            cutoff=True,
        )

    def _make_tracing_inputs(self, n: int):
        return [{"forward": (torch.randn(5, 1),)} for _ in range(n)]


class BesselBasis(nn.Module):
    def __init__(self, r_max, num_basis=8, trainable=True):
        r"""Radial Bessel Basis, as proposed in
            DimeNet: https://arxiv.org/abs/2003.03123

        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(
            self.bessel_weights * x.unsqueeze(-1) / self.r_max  # noqa: E501
        )

        return self.prefactor * (numerator / x.unsqueeze(-1))


@compile_mode("script")
class SmoothBesselBasis(nn.Module):
    def __init__(self, r_max, max_n=10):
        r"""Smooth Radial Bessel Basis, as proposed
            in DimeNet: https://arxiv.org/abs/2003.03123
            This is an orthogonal basis with first
            and second derivative at the cutoff
            equals to zero. The function was derived from
            the order 0 spherical Bessel function,
            and was expanded by the different zero roots
        Ref:
            https://arxiv.org/pdf/1907.02374.pdf
        Args:
            r_max: torch.Tensor distance tensor
            max_n: int, max number of basis, expanded by the zero roots
        Returns: expanded spherical harmonics with
                 derivatives smooth at boundary
        Parameters
        ----------
        """
        super(SmoothBesselBasis, self).__init__()
        self.max_n = max_n
        n = torch.arange(0, max_n).float()[None, :]
        PI = 3.1415926535897
        SQRT2 = 1.41421356237
        fnr = (
            (-1) ** n
            * SQRT2
            * PI
            / r_max**1.5
            * (n + 1)
            * (n + 2)
            / torch.sqrt(2 * n**2 + 6 * n + 5)
        )
        en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = [torch.tensor(1.0).float()]
        for i in range(1, max_n):
            dn.append(1 - en[0, i] / dn[-1])
        dn = torch.stack(dn)
        self.register_buffer("dn", dn)
        self.register_buffer("en", en)
        self.register_buffer("fnr_weights", fnr)
        self.register_buffer(
            "n_1_pi_cutoff",
            ((torch.arange(0, max_n).float() + 1) * PI / r_max).reshape(1, -1),
        )
        self.register_buffer(
            "n_2_pi_cutoff",
            ((torch.arange(0, max_n).float() + 2) * PI / r_max).reshape(1, -1),
        )
        self.register_buffer("r_max", torch.tensor(r_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Smooth Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        x_1 = x.unsqueeze(-1) * self.n_1_pi_cutoff
        x_2 = x.unsqueeze(-1) * self.n_2_pi_cutoff
        fnr = self.fnr_weights * (torch.sin(x_1) / x_1 + torch.sin(x_2) / x_2)
        gn = [fnr[:, 0]]
        for i in range(1, self.max_n):
            gn.append(
                1
                / torch.sqrt(self.dn[i])
                * (
                    fnr[:, i]
                    + torch.sqrt(self.en[0, i] / self.dn[i - 1]) * gn[-1]  # noqa: E501
                )
            )
        return torch.transpose(torch.stack(gn), 1, 0)


# class GaussianBasis(nn.Module):
#     r_max: float

#     def __init__(self, r_max, r_min=0.0, num_basis=8, trainable=True):
#         super().__init__()

#         self.trainable = trainable
#         self.num_basis = num_basis

#         self.r_max = float(r_max)
#         self.r_min = float(r_min)

#         means = torch.linsspace(self.r_min, self.r_max, self.num_basis)
#         stds = torch.full(size=means.size, fill_value=means[1] - means[0])
#         if self.trainable:
#             self.means = nn.Parameter(means)
#             self.stds = nn.Parameter(stds)
#         else:
#             self.register_buffer("means", means)
#             self.register_buffer("stds", stds)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = (x[..., None] - self.means) / self.stds
#         x = x.square().mul(-0.5).exp() / self.stds  # sqrt(2 * pi)
@torch.jit.script
def _spherical_harmonics(lmax: int, x: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x) * 0.5 * math.sqrt(1.0 / math.pi)
    if lmax == 0:
        return torch.stack(
            [
                sh_0_0,
            ],
            dim=-1,
        )

    sh_1_1 = math.sqrt(3.0 / (4.0 * math.pi)) * x
    if lmax == 1:
        return torch.stack([sh_0_0, sh_1_1], dim=-1)

    sh_2_2 = math.sqrt(5.0 / (16.0 * math.pi)) * (3.0 * x**2 - 1.0)
    if lmax == 2:
        return torch.stack([sh_0_0, sh_1_1, sh_2_2], dim=-1)

    sh_3_3 = math.sqrt(7.0 / (16.0 * math.pi)) * x * (5.0 * x**2 - 3.0)
    if lmax == 3:
        return torch.stack([sh_0_0, sh_1_1, sh_2_2, sh_3_3], dim=-1)

    raise ValueError("lmax must be less than 8")


class SphericalBasisLayer(nn.Module):
    def __init__(self, max_n, max_l, cutoff):
        super(SphericalBasisLayer, self).__init__()

        assert max_l <= 4, "lmax must be less than 5"
        assert max_n <= 4, "max_n must be less than 5"

        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = cutoff

        # retrieve formulas
        self.register_buffer(
            "factor", torch.sqrt(torch.tensor(2.0 / (self.cutoff**3)))
        )
        coef = torch.zeros(4, 9, 4)
        coef[0, 0, :] = torch.tensor(
            [
                3.14159274101257,
                6.28318548202515,
                9.42477798461914,
                12.5663709640503,
            ]  # noqa: E501
        )
        coef[1, :4, :] = torch.tensor(
            [
                [
                    -1.02446483277785,
                    -1.00834335996107,
                    -1.00419641763893,
                    -1.00252381898662,
                ],
                [
                    4.49340963363647,
                    7.7252516746521,
                    10.9041213989258,
                    14.0661935806274,
                ],  # noqa: E501
                [
                    0.22799275301076,
                    0.130525632358311,
                    0.092093290316619,
                    0.0712718627992818,
                ],
                [
                    4.49340963363647,
                    7.7252516746521,
                    10.9041213989258,
                    14.0661935806274,
                ],  # noqa: E501
            ]
        )
        coef[2, :6, :] = torch.tensor(
            [
                [
                    -1.04807944170731,
                    -1.01861796359391,
                    -1.01002272174988,
                    -1.00628955560036,
                ],
                [
                    5.76345920562744,
                    9.09501171112061,
                    12.322940826416,
                    15.5146026611328,
                ],  # noqa: E501
                [
                    0.545547077361439,
                    0.335992298618515,
                    0.245888396928293,
                    0.194582402961821,
                ],
                [
                    5.76345920562744,
                    9.09501171112061,
                    12.322940826416,
                    15.5146026611328,
                ],  # noqa: E501
                [
                    0.0946561878721665,
                    0.0369424811413594,
                    0.0199537107571916,
                    0.0125418876146463,
                ],
                [
                    5.76345920562744,
                    9.09501171112061,
                    12.322940826416,
                    15.5146026611328,
                ],  # noqa: E501
            ]
        )
        coef[3, :8, :] = torch.tensor(
            [
                [
                    1.06942831392075,
                    1.0292173312802,
                    1.01650804843248,
                    1.01069656069999,
                ],  # noqa: E501
                [
                    6.9879322052002,
                    10.4171180725098,
                    13.6980228424072,
                    16.9236221313477,
                ],  # noqa: E501
                [
                    0.918235852195231,
                    0.592803493701152,
                    0.445250264272671,
                    0.358326327374518,
                ],
                [
                    6.9879322052002,
                    10.4171180725098,
                    13.6980228424072,
                    16.9236221313477,
                ],  # noqa: E501
                [
                    0.328507713452024,
                    0.142266673367543,
                    0.0812617757677838,
                    0.0529328657590962,
                ],
                [
                    6.9879322052002,
                    10.4171180725098,
                    13.6980228424072,
                    16.9236221313477,
                ],  # noqa: E501
                [
                    0.0470107184508114,
                    0.0136570088173405,
                    0.0059323726279831,
                    0.00312775039221944,
                ],
                [
                    6.9879322052002,
                    10.4171180725098,
                    13.6980228424072,
                    16.9236221313477,
                ],  # noqa: E501
            ]
        )
        self.register_buffer("coef", coef)

    def forward(self, r, theta_val):
        r = r / self.cutoff
        # Denote empty lists for rbf and cbf
        rbfs = []

        for j in range(self.max_l):
            rbfs.append(torch.sin(self.coef[0, 0, j] * r) / r)

        if self.max_n > 1:
            for j in range(self.max_l):
                rbfs.append(
                    (
                        self.coef[1, 0, j]
                        * r
                        * torch.cos(self.coef[1, 1, j] * r)  # noqa: E501
                        + self.coef[1, 2, j]
                        * torch.sin(self.coef[1, 3, j] * r)  # noqa: E501
                    )
                    / r**2
                )

            if self.max_n > 2:
                for j in range(self.max_l):
                    rbfs.append(
                        (
                            self.coef[2, 0, j]
                            * (r**2)
                            * torch.sin(self.coef[2, 1, j] * r)
                            - self.coef[2, 2, j]
                            * r
                            * torch.cos(self.coef[2, 3, j] * r)  # noqa: E501
                            + self.coef[2, 4, j]
                            * torch.sin(self.coef[2, 5, j] * r)  # noqa: E501
                        )
                        / (r**3)
                    )

                if self.max_n > 3:
                    for j in range(self.max_l):
                        rbfs.append(
                            (
                                self.coef[3, 0, j]
                                * (r**3)
                                * torch.cos(self.coef[3, 1, j] * r)
                                - self.coef[3, 2, j]
                                * (r**2)
                                * torch.sin(self.coef[3, 3, j] * r)
                                - self.coef[3, 4, j]
                                * r
                                * torch.cos(self.coef[3, 5, j] * r)
                                + self.coef[3, 6, j]
                                * torch.sin(self.coef[3, 7, j] * r)  # noqa: E501
                            )
                            / r**4
                        )

        rbfs = torch.stack(rbfs, dim=-1)
        rbfs = rbfs * self.factor

        cbfs = _spherical_harmonics(self.max_l - 1, torch.cos(theta_val))
        cbfs = cbfs.repeat_interleave(self.max_n, dim=1)

        return rbfs * cbfs
