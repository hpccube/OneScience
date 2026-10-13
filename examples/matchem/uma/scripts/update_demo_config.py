#!/usr/bin/env python3
"""用 create_uma_finetune_dataset.py 生成的 data yaml 更新 demo 配置文件。"""

import argparse
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def update_demo_config(demo_config_path: Path, generated_data_path: Path):
    demo_cfg = load_yaml(demo_config_path)
    gen_data = load_yaml(generated_data_path)

    # 生成的 yaml 直接就是 data 部分的内容（没有顶层 data 键）
    demo_cfg["data"] = gen_data

    # 保持 train/val 路径使用 ONESCIENCE_DATASETS_DIR 变量（数据放在 UMA/data/ 下）
    demo_cfg["data"]["train_dataset"]["splits"]["train"][
        "src"
    ] = "${ONESCIENCE_DATASETS_DIR}/data/oc20_finetune/train"
    demo_cfg["data"]["val_dataset"]["splits"]["val"][
        "src"
    ] = "${ONESCIENCE_DATASETS_DIR}/data/oc20_finetune/val"

    save_yaml(demo_config_path, demo_cfg)
    print(f"[OK] 已更新 {demo_config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-config", type=Path, required=True)
    parser.add_argument(
        "--generated-data",
        type=Path,
        default=Path("data/oc20_finetune/data/uma_conserving_data_task_energy_force.yaml"),
    )
    args = parser.parse_args()
    update_demo_config(args.demo_config, args.generated_data)
