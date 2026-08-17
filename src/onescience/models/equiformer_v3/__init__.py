"""Equiformer V3 atomistic models adapted to OneScience."""

from .equiformer_v3 import EquiformerV3_OC
from .equiformer_v3_dens import EquiformerV3DeNS_OC

EquiformerV3 = EquiformerV3_OC
EquiformerV3DeNS = EquiformerV3DeNS_OC

__all__ = [
    "EquiformerV3",
    "EquiformerV3DeNS",
    "EquiformerV3DeNS_OC",
    "EquiformerV3_OC",
]
