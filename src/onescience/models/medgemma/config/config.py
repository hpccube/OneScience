# MedGemma 配置管理器
# 重用 Protenix 的 ConfigManager

import os
from collections.abc import Mapping

from ml_collections.config_dict import ConfigDict

from onescience.models.protenix.config.config import (
    ConfigManager,
    parse_configs as protenix_parse_configs,
    parse_sys_args,
    load_config as protenix_load_config,
    save_config,
)

__all__ = [
    "ConfigManager",
    "parse_configs",
    "parse_sys_args",
    "load_config",
    "save_config",
]


def parse_configs(base_configs: dict, sys_args=None, fill_required_with_null: bool = False):
    """
    解析 MedGemma 配置

    Args:
        base_configs: 基础配置字典
        sys_args: 命令行参数（可选）
        fill_required_with_null: 是否用 None 填充必需值

    Returns:
        ConfigDict: 解析后的配置
    """
    return protenix_parse_configs(base_configs, sys_args, fill_required_with_null)


def load_config(path: str) -> ConfigDict:
    """Load a MedGemma YAML config as a ConfigDict for attribute access."""
    return ConfigDict(_expand_path_vars(protenix_load_config(path)))


def _expand_path_vars(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, Mapping):
        return {key: _expand_path_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_path_vars(item) for item in value]
    return value
