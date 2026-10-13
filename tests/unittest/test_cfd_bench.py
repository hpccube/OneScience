from contextlib import redirect_stdout
from io import StringIO

import torch

from onescience.models.cfdbench.auto_deeponet import AutoDeepONet
from onescience.models.cfdbench.auto_deeponet_cnn import AutoDeepONetCnn
from onescience.models.cfdbench.auto_edeeponet import AutoEDeepONet
from onescience.models.cfdbench.auto_ffn import AutoFfn
from onescience.models.cfdbench.deeponet import DeepONet
from onescience.models.cfdbench.ffn import FfnModel
from onescience.models.cfdbench.fno.fno2d import Fno2d
from onescience.models.cfdbench.loss import MseLoss
from onescience.models.cfdbench.resnet import ResNet
from onescience.models.cfdbench.unet import UNet


def print_shape(name, out, target):
    print(f"Function: {name} Forward")
    print(f"output shape: {out.shape}")
    print(f"target shape: {target}\n")


def print_num_params(name, model):
    num_params = sum(param.numel() for param in model.parameters())
    print(f"{name} parameters: {num_params}")


torch.manual_seed(0)

loss = MseLoss(normalize=False)
case_params = torch.randn(1, 2)
t = torch.tensor([[0.1]])
inputs = torch.randn(1, 2, 16, 16)
mask = torch.ones(1, 16, 16)
query_idxs = torch.tensor([[0, 0], [1, 2], [3, 4], [5, 6]], dtype=torch.long)

dense_models = [
    (
        "CFDBench FFN",
        FfnModel(loss, widths=[5, 8, 1], num_label_samples=4),
        (case_params, t),
        {"query_idxs": query_idxs},
        torch.Size([1, 4]),
    ),
    (
        "CFDBench DeepONet",
        DeepONet(2, 3, loss, num_label_samples=4, branch_depth=2, trunk_depth=2, width=8),
        (case_params, t),
        {"query_idxs": query_idxs},
        torch.Size([1, 4]),
    ),
]

for name, model, args, kwargs, target in dense_models:
    print_num_params(name, model)
    model.eval()
    with torch.no_grad():
        out = model(*args, **kwargs)["preds"]
    assert out.shape == target, f"{name}: got {out.shape}, expected {target}"
    print_shape(name, out, target)

with redirect_stdout(StringIO()):
    auto_edeep_onet = AutoEDeepONet(
        16 * 16,
        2,
        2,
        loss,
        num_label_samples=4,
        branch_depth=2,
        trunk_depth=2,
        width=8,
    )

auto_models = [
    (
        "CFDBench AutoFfn",
        AutoFfn(16 * 16, 2, 2, loss, num_label_samples=4, depth=1, width=8),
        inputs,
        torch.Size([1, 1, 16, 16]),
    ),
    (
        "CFDBench AutoDeepONet",
        AutoDeepONet(16 * 16 + 2, 2, loss, num_label_samples=4, branch_depth=2, trunk_depth=2, width=8),
        inputs,
        torch.Size([1, 1, 16, 16]),
    ),
    (
        "CFDBench AutoEDeepONet",
        auto_edeep_onet,
        inputs,
        torch.Size([1, 16 * 16]),
    ),
    (
        "CFDBench FNO2d",
        Fno2d(2, 2, 2, loss, num_layers=1, modes1=4, modes2=4, hidden_dim=8),
        inputs,
        torch.Size([1, 2, 16, 16]),
    ),
    (
        "CFDBench UNet",
        UNet(2, 2, loss, n_case_params=2, insert_case_params_at="hidden", dim=4),
        inputs,
        torch.Size([1, 2, 16, 16]),
    ),
    (
        "CFDBench ResNet",
        ResNet(2, 2, 2, loss, hidden_chan=4, num_blocks=1, kernel_size=3, padding=1),
        inputs,
        torch.Size([1, 2, 16, 16]),
    ),
]

for name, model, model_inputs, target in auto_models:
    print_num_params(name, model)
    model.eval()
    with torch.no_grad():
        out = model(model_inputs, case_params, mask=mask)["preds"]
    assert out.shape == target, f"{name}: got {out.shape}, expected {target}"
    print_shape(name, out, target)

cnn = AutoDeepONetCnn(2, 2, loss, height=64, width=64, num_case_params=2, trunk_depth=1)
print_num_params("CFDBench AutoDeepONetCnn", cnn)
cnn.eval()
with torch.no_grad():
    out = cnn(torch.randn(1, 2, 64, 64), case_params, mask=torch.ones(1, 64, 64))[
        "preds"
    ]
target = torch.Size([1, 1, 64, 64])
assert out.shape == target, f"CFDBench AutoDeepONetCnn: got {out.shape}, expected {target}"
print_shape("CFDBench AutoDeepONetCnn", out, target)
