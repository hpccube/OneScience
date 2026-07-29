"""Parameter-level exponential moving average used by Boltz-1."""

import torch


class ExponentialMovingAverage:
    """Maintain an exponential moving average of trainable parameters."""

    def __init__(self, parameters, decay, use_num_updates=True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for shadow, parameter in zip(self.shadow_params, parameters):
                shadow.sub_(one_minus_decay * (shadow - parameter))

    def compatible(self, parameters):
        if len(self.shadow_params) != len(parameters):
            print(
                f"Model has {len(self.shadow_params)} parameter tensors, "
                f"the incoming ema {len(parameters)}"
            )
            return False
        for shadow, parameter in zip(self.shadow_params, parameters):
            if parameter.data.shape != shadow.data.shape:
                print(
                    f"Model has parameter tensor of shape {shadow.data.shape} , "
                    f"the incoming ema {parameter.data.shape}"
                )
                return False
        return True

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for shadow, parameter in zip(self.shadow_params, parameters):
            if parameter.requires_grad:
                parameter.data.copy_(shadow.data)

    def store(self, parameters):
        self.collected_params = [parameter.clone() for parameter in parameters]

    def restore(self, parameters):
        for collected, parameter in zip(self.collected_params, parameters):
            parameter.data.copy_(collected.data)

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = [tensor.to(device) for tensor in state_dict["shadow_params"]]

    def to(self, device):
        self.shadow_params = [tensor.to(device) for tensor in self.shadow_params]


# Preserve the historical pickle path while the implementation lives under optim.
ExponentialMovingAverage.__module__ = "onescience.utils.boltz.model"
