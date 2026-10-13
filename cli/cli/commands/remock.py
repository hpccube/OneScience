import click
from pathlib import Path
from ..core.runner import run_model
from ..core.registry import model_registry


@click.command("remock")
@click.argument("model_alias")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="跳过确认提示，直接执行重置")
def remock(model_alias: str, yes: bool) -> None:
    """重置模型环境（删除 result/、checkpoints/、logs/、__pycache__/ 等可重建目录）"""
    info = model_registry.resolve(model_alias)
    if not info:
        click.secho(f"未知模型: {model_alias}", fg="red")
        raise SystemExit(1)

    model_dir = info.get("model_dir")
    if not model_dir:
        click.secho(f"无法获取模型目录: {model_alias}", fg="red")
        raise SystemExit(1)

    model_dir = Path(model_dir)
    sub_model = info.get("sub_model", "")

    # 路径安全性校验
    if _is_dangerous_path(model_dir):
        click.secho(
            f"危险路径，已拒绝执行: {model_dir}\n"
            f"  模型目录不能是系统根目录或用户主目录",
            fg="red",
        )
        raise SystemExit(1)

    if not model_dir.exists():
        click.secho(f"模型目录不存在: {model_dir}", fg="red")
        raise SystemExit(1)

    if not yes:
        if sub_model:
            click.secho(
                f"即将重置模型 '{model_alias}'（{sub_model}）的环境:\n"
                f"  目录: {model_dir}\n"
                f"  - {sub_model}/ 子目录下的 result/  results/  checkpoints/  logs/\n"
                f"  - result/  results/  checkpoints/  logs/ 中匹配 '{sub_model}' 的内容\n"
                f"  - 所有 __pycache__/ 目录\n"
                f"  - *_execution.log 文件",
                fg="yellow",
            )
        else:
            click.secho(
                f"即将重置模型 '{model_alias}' 的环境:\n"
                f"  目录: {model_dir}\n"
                f"  - result/  results/  checkpoints/  logs/\n"
                f"  - 所有 __pycache__/ 目录\n"
                f"  - *_execution.log 文件",
                fg="yellow",
            )
        click.confirm("确认继续?", abort=True)

    r = run_model(model_alias, "remock", "")
    if r["success"]:
        click.secho("环境重置完成", fg="green")
        raise SystemExit(0)
    else:
        click.secho(f"重置失败: {r.get('error', '')}", fg="red")
        raise SystemExit(1)


def _is_dangerous_path(path: Path) -> bool:
    """检查路径是否为危险路径（根目录、用户主目录等）"""
    resolved = path.resolve()
    # 根目录（Windows C:\ 或 POSIX /）的 parent 指向自身
    if resolved.parent == resolved:
        return True
    # 用户主目录
    home = Path.home().resolve()
    if resolved == home:
        return True
    return False
