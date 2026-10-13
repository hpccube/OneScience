import warnings

import torch

from onescience.models.deepcfd.AutoEncoder import AutoEncoder
from onescience.models.deepcfd.AutoEncoderEx import AutoEncoderEx
from onescience.models.deepcfd.UNet import UNet
from onescience.models.deepcfd.UNetEx import UNetEx


warnings.filterwarnings(
    "ignore", message="`torch.nn.utils.weight_norm` is deprecated.*"
)
torch.manual_seed(0)

x = torch.randn(1, 3, 32, 32)
models = [
    ("DeepCFD AutoEncoder", AutoEncoder(3, 2, filters=[4, 8])),
    ("DeepCFD AutoEncoderEx", AutoEncoderEx(3, 2, filters=[4, 8])),
    ("DeepCFD UNet", UNet(3, 2, base_channels=4, num_stages=2, normtype="bn")),
    ("DeepCFD UNetEx", UNetEx(3, 2, base_channels=4, num_stages=2, normtype="bn")),
]

for name, model in models:
    num_params = sum(param.numel() for param in model.parameters())
    print(f"{name} parameters: {num_params}")
    model.eval()
    with torch.no_grad():
        out = model(x)
    target = torch.Size([1, 2, 32, 32])
    assert out.shape == target, f"{name}: got {out.shape}, expected {target}"
    print(f"Function: {name} Forward")
    print(f"output shape: {out.shape}")
    print(f"target shape: {target}\n")
