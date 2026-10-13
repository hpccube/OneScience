import typing as t
import click
from pathlib import Path
from ..core.runner import run_model, collect_results, print_comparison, print_metrics, print_perf_comparison
from ..core.registry import model_registry


# 各领域的默认数据集
_DOMAIN_DEFAULT_DATASETS = {
    "earth": "era5",
    "cfd": "airfoil",
    "biosciences": "evo2",
    "matchem": "mace",
}

def _get_model_default_dataset(model_dir: Path) -> str | None:
    """从模型目录的 conf/config.yaml 中读取 datapipe.name，转小写作为数据集名"""
    config_path = model_dir / "conf" / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None
        pipe_name = data.get("datapipe", {}).get("name", "")
        if not pipe_name:
            return None
        return pipe_name.lower()
    except Exception:
        return None


def _is_container_alias(alias: str, all_models: list[dict]) -> bool:
    """检查别名是否为容器模型

    容器模型（如 cfd_benchmark）的 sub_model 为空，但该目录下有其他模型
    以非空 sub_model 独立注册。执行时不应重复执行容器模型本身。
    """
    info = next((m for m in all_models if m["alias"] == alias), None)
    if not info:
        return False
    if info.get("sub_model", ""):
        return False
    model_name = info.get("model", "")
    if not model_name:
        return False
    return any(
        m["alias"] != alias and m.get("model") == model_name and m.get("sub_model", "")
        for m in all_models
    )


def _resolve_model_dataset(alias: str, info: dict, user_dataset: str | None = None) -> str | None:
    """解析模型的有效数据集名称

    优先级:
      1. 用户指定的 -dataset（最高优先级，信任用户选择）
      2. 模型 conf/config.yaml 中声明的 datapipe.name
      3. 与模型同名的内置数据集（BUILTIN_DATASETS 中存在相同 key）
      4. 该模型所属领域的默认数据集（_DOMAIN_DEFAULT_DATASETS）

    Returns:
        数据集名称（str），或 None（完全无法确定时）
    """
    # 1. 用户指定
    if user_dataset:
        return user_dataset

    # 2. 模型配置文件声明
    model_dir = info.get("model_dir")
    if model_dir:
        model_ds = _get_model_default_dataset(Path(model_dir))
        if model_ds:
            return model_ds

    # 3. 同名数据集
    from ..core.config import BUILTIN_DATASETS
    if alias.lower() in BUILTIN_DATASETS:
        return alias.lower()

    # 4. 领域默认
    domain = info.get("domain", "")
    return _DOMAIN_DEFAULT_DATASETS.get(domain)


@click.command("bench")
@click.option("-dataset", default=None, help="数据集名称或路径（不指定则自动检测）")
@click.option("-models", default=None, help="模型别名列表，逗号分隔")
@click.option("--domain", default=None, help="按领域执行所有模型（如 earth/cfd/all）")
@click.option("--dir", "model_dir", default=None, help="按模型目录名执行该目录下所有模型（如 CFD_Benchmark）")
@click.option("--epoch", default=None, type=int,
              help="统一设置训练轮数（所有领域通用，详见 onescience help）")
@click.option("-O", "overrides", multiple=True, default=None,
              help="覆写 config.yaml 任意参数，支持点号路径，可多次使用（执行后自动还原）")
def bench(dataset, models, domain, model_dir, epoch, overrides):
    """使用指定数据集运行多个模型（训练+推理+评估）"""

    # ── 解析模型列表 ──────────────────────────────────
    aliases = []
    if models:
        aliases = [m.strip() for m in models.split(",") if m.strip()]
    elif domain:
        all_models = model_registry.list_models()
        if domain == "all":
            aliases = [m["alias"] for m in all_models]
        else:
            aliases = [m["alias"] for m in all_models if m.get("domain") == domain]
        # 隐式选择时，排除容器模型（如 cfd_benchmark 已由子模型覆盖）
        aliases = [a for a in aliases if not _is_container_alias(a, all_models)]
        if not aliases:
            click.secho(f"领域 '{domain}' 下没有找到可用模型", fg="red")
            return
    elif model_dir:
        all_models = model_registry.list_models()
        aliases = [m["alias"] for m in all_models if m.get("model") == model_dir]
        # 隐式选择时，排除容器模型（如 cfd_benchmark 已由子模型覆盖）
        aliases = [a for a in aliases if not _is_container_alias(a, all_models)]
        if not aliases:
            click.secho(f"目录 '{model_dir}' 下没有找到可用模型", fg="red")
            return
    else:
        click.secho("请指定 -models / --domain / --dir 参数", fg="red")
        return

    # ── 执行模型 ──────────────────────────────────
    results = []

    for alias in aliases:
        info = model_registry.resolve(alias)
        if not info:
            click.secho(f"  跳过 {alias}（未知模型）", fg="yellow")
            continue

        effective_ds = _resolve_model_dataset(alias, info, user_dataset=dataset)
        if not effective_ds:
            domain_name = info.get("domain", "?")
            click.secho(
                f"  跳过 {alias}（领域 '{domain_name}' 无默认数据集，"
                f"模型未声明 datapipe.name，且无同名内置数据集）",
                fg="yellow",
            )
            continue

        _run_single_model(info, alias, effective_ds, results, epoch=epoch, overrides=overrides)

    collect_results(results)
    click.echo(f"所有模型执行完成")
    print_comparison(results)
    print_perf_comparison(results)


def _run_single_model(info: dict, alias: str, dataset: str, results: list,
                      epoch: t.Optional[int] = None,
                      overrides: t.Optional[t.List[str]] = None):
    """执行单个模型并记录结果"""
    # info 由调用方传入，避免重复 resolve() 触发不必要的下载
    click.echo(f"\n{'=' * 48}")
    click.secho(f"开始执行模型: {alias}", fg="green")
    click.secho(f"模型领域: {info['domain']}", fg="green")
    click.secho(f"数据集: {dataset}", fg="green")
    if epoch is not None:
        click.secho(f"训练轮数: {epoch}", fg="green")
    click.echo(f"{'=' * 48}")
    r = run_model(alias, "bench", dataset, epoch=epoch, overrides=overrides)
    results.append(r)
    # 每个模型执行后尝试释放 GPU 显存，避免累积导致 OOM
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if r["success"]:
        click.secho(f"模型执行完成: {alias}", fg="green")
    else:
        err_msg = r.get("error") or r.get("output", "")
        click.secho(f"模型执行失败: {err_msg}", fg="red")
    print_metrics(r)
    click.echo()
