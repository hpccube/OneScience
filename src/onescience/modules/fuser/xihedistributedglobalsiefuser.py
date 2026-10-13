import torch
import torch.nn as nn
from onescience.modules.func_utils import Mlp
from onescience.modules.attention.xihedistributedfeaturegroupattention import DistributedFeatureGroupingAttention
from onescience.modules.attention.xihedistributedfeatureungroupattention import DistributedFeatureUngroupingAttention
from onescience.modules.mlp.xihedistributedmlp import XiheDistributedMlp


class XiheDistributedGlobalSIEFuser(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=True,
        num_groups=32,
        norm_layer=nn.LayerNorm,
        config=None,
    ):
        super().__init__()
        self.dim = dim

        self.feature_grouping = DistributedFeatureGroupingAttention(
            dim=dim,
            num_heads=num_heads,
            num_groups=num_groups,
            config=config,
        )
        self.feature_ungrouping = DistributedFeatureUngroupingAttention(
            dim=dim,
            num_heads=num_heads,
            config=config,
        )

        self.group_propagation = XiheDistributedMlp(
            dim=dim,
            num_groups=num_groups,
            config=config,
        )

    def forward(self, obj):
        if isinstance(obj, dict):
            x = obj["x"]
            mask = obj.get("mask")
            if mask is not None:
                mask = mask.clone().detach().float()
            obj["y"] = x
        else:
            x = obj.x
            mask = getattr(obj, 'mask', None)
            obj.y = x
            obj = {"x": x, "mask": mask, "y": x}

        x = self.feature_grouping(obj, mask=mask)
        x = self.group_propagation(x)
        obj["x"] = x
        x = self.feature_ungrouping(obj, mask=mask)

        return x
