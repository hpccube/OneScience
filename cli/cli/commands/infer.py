import click
from ..core.runner import run_model, print_metrics
from ..core.registry import model_registry


@click.command("infer")
@click.argument("model_alias")
@click.option("-dataset", required=True, help="数据集名称或路径")
@click.option("--epoch", default=None, type=int,
              help="统一设置训练轮数（所有领域通用，详见 onescience help）")
@click.option("-O", "overrides", multiple=True, default=None,
              help="覆写 config.yaml 任意参数，支持点号路径，可多次使用")
def infer(model_alias, dataset, epoch, overrides):
    """仅执行模型推理"""
    info = model_registry.resolve(model_alias)
    if not info:
        click.secho(f"未知模型: {model_alias}", fg="red")
        return
    click.secho(f"开始推理: {model_alias}", fg="green")
    click.secho(f"数据集: {dataset}", fg="green")
    if epoch is not None:
        click.secho(f"训练轮数: {epoch}", fg="green")
    r = run_model(model_alias, "infer", dataset, epoch=epoch, overrides=overrides)
    if r["success"]:
        click.secho(f"推理完成: {model_alias}", fg="green")
    else:
        err_msg = r.get("error") or r.get("output", "")
        click.secho(f"推理失败: {err_msg}", fg="red")
    print_metrics(r)
