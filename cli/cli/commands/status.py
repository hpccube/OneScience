import click
from pathlib import Path
from ..core.registry import model_registry, EXAMPLES_DIR
from ..core.config import BUILTIN_DOMAIN_DIR_MAP


@click.command("status")
@click.argument("model_aliases", required=False)
def status(model_aliases):
    """查看模型执行状态（只读，不触发自动下载）"""
    models = model_registry.list_models()
    if model_aliases:
        aliases = [a.strip() for a in model_aliases.split(",") if a.strip()]
        models = [m for m in models if m["alias"] in aliases]
    click.secho("模型执行状态", fg="green")
    for info in models:
        alias = info["alias"]
        sub_model = info.get("sub_model", "")

        # 构建模型目录路径，兼容领域目录映射
        model_dir = _build_model_dir(info)
        if not model_dir or not model_dir.exists():
            click.echo(f"  {alias:<20} 目录不存在")
            continue

        # 统计日志：共享目录下只匹配当前子模型，避免混入其他模型日志
        if sub_model:
            log_count = len(list(model_dir.glob(f"*{sub_model}*.log")))
            log_count += len(list(model_dir.glob(f"*{sub_model}*.log.*")))
        else:
            log_count = len(list(model_dir.glob("*.log")))
            log_count += len(list(model_dir.glob("*.log.*")))

        # 检查常见产物目录
        result_count = 0
        for d in ["result", "results", "checkpoints"]:
            p = model_dir / d
            if p.exists():
                # 共享目录下只统计子模型命名的内容
                if sub_model:
                    sub_p = p / sub_model
                    if sub_p.exists():
                        result_count += len(list(sub_p.rglob("*")))
                    result_count += len(list(p.glob(f"*{sub_model}*")))
                else:
                    result_count += len(list(p.rglob("*")))

        if log_count > 0 and result_count > 0:
            status_str = "已完成"
        elif log_count > 0:
            status_str = "已运行"
        else:
            status_str = "未运行"
        click.echo(f"  {alias:<20} {status_str:<8} {log_count}个日志  {result_count}个结果")


def _build_model_dir(info: dict) -> Path:
    """根据模型信息构建目录路径"""
    domain = info.get("domain", "")
    model = info.get("model", "")
    source = info.get("source", "")

    if source == "builtin":
        dir_name = BUILTIN_DOMAIN_DIR_MAP.get(domain, domain)
        return EXAMPLES_DIR / dir_name / model
    elif source == "scan":
        # 扫描发现的模型，尝试在 examples 下或当前目录定位
        p = EXAMPLES_DIR / model
        if p.exists():
            return p
        return Path(model) if Path(model).exists() else None
    else:
        # 自定义模型等，按 domain/model 路径尝试
        return EXAMPLES_DIR / domain / model
