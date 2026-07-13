import typing as t
import click
from ..core.runner import run_model, print_metrics
from ..core.registry import model_registry


@click.command("train")
@click.argument("model_alias")
@click.option("-dataset", default=None, help="数据集名称或路径（部分模型可省略，使用模型自带配置）")
@click.option("--epoch", default=None, type=int,
              help="统一设置训练轮数（所有领域通用，详见 onescience help）")
@click.option("-O", "overrides", multiple=True, default=None,
              help="覆写 config.yaml 任意参数，支持点号路径，可多次使用（执行后自动还原）")
def train(model_alias, dataset, epoch, overrides):
    """仅执行模型训练"""
    info = model_registry.resolve(model_alias)
    if not info:
        click.secho(f"未知模型: {model_alias}", fg="red")
        return
    click.secho(f"开始训练模型: {model_alias}", fg="green")
    if dataset:
        click.secho(f"数据集: {dataset}", fg="green")
    if epoch is not None:
        click.secho(f"训练轮数: {epoch}", fg="green")
    r = run_model(model_alias, "TRAIN", dataset, epoch=epoch, overrides=overrides)
    if r["success"]:
        click.secho(f"训练完成: {model_alias}", fg="green")
    else:
        err_msg = r.get("error") or r.get("output", "")
        click.secho(f"训练失败: {err_msg}", fg="red")
    print_metrics(r)
