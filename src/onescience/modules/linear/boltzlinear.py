"""Linear primitives used by Boltz modules."""

from functools import partial

from torch.nn import Linear


LinearNoBias = partial(Linear, bias=False)


__all__ = ["LinearNoBias"]
