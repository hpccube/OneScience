import copy
import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from onescience.models.protenix.config import parse_configs
from onescience.models.protenix.protenix import Protenix


REQUIRED_ARG_STR = (
    "--project protenix_unittest "
    "--run_name protenix_parameter_count "
    "--base_dir ./outputs "
    "--eval_interval 1 "
    "--log_interval 1 "
    "--max_steps 1"
)

MINIMAL_DATA_CONFIGS = {
    "msa": {
        "enable": False,
        "strategy": "random",
        "merge_method": "dense_max",
        "min_size": {
            "train": 1,
            "test": 1,
        },
        "sample_cutoff": {
            "train": 1,
            "test": 1,
        },
    },
    "template": {
        "enable": False,
    },
}


def parameter_count(model):
    return sum(param.numel() for param in model.parameters())


def trainable_parameter_count(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def assert_nonzero_parameters(model):
    total = parameter_count(model)
    trainable = trainable_parameter_count(model)
    assert total > 0, "model should have parameters"
    assert trainable > 0, "model should have trainable parameters"
    return total


def load_configs_base():
    config_path = (
        REPO_ROOT
        / "examples"
        / "biosciences"
        / "protenix"
        / "configs"
        / "configs_base.py"
    )
    spec = importlib.util.spec_from_file_location("protenix_configs_base", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return copy.deepcopy(module.configs)


def build_protenix_configs():
    configs = load_configs_base()
    configs["use_deepspeed_evo_attention"] = False
    configs["data"] = MINIMAL_DATA_CONFIGS
    return parse_configs(configs=configs, arg_str=REQUIRED_ARG_STR)


def main():
    torch.manual_seed(0)
    configs = build_protenix_configs()
    model = Protenix(configs)
    total_params = assert_nonzero_parameters(model)
    trainable_params = trainable_parameter_count(model)

    print("Function: Protenix Model Parameter Count")
    print(f"parameter count: {total_params:,}")
    print(f"trainable parameter count: {trainable_params:,}")
    print(f"parameter count (M): {total_params / 1_000_000:.3f}")


if __name__ == "__main__":
    main()
else:
    main()
