import os
import sys
import shutil
import subprocess
from pathlib import Path

import click
from ..core.registry import model_registry, get_model_dir


@click.group("deploy")
def deploy_group():
    """模型部署（ONNX 导出与推理服务）"""


def _find_checkpoints(model_dir: Path) -> list[Path]:
    """扫描模型目录下的 checkpoint 文件"""
    checkpoints = []

    # checkpoints/ 目录
    ckpt_dir = model_dir / "checkpoints"
    if ckpt_dir.is_dir():
        checkpoints.extend(sorted(ckpt_dir.glob("*.pth")))
        checkpoints.extend(sorted(ckpt_dir.glob("*.pt")))

    # 模型目录根目录
    checkpoints.extend(sorted(model_dir.glob("*.pth")))
    checkpoints.extend(sorted(model_dir.glob("*.pt")))

    return checkpoints


def _find_model_script(model_dir: Path) -> Path | None:
    """在模型目录中查找可用的 Python 入口脚本"""
    for pattern in ("train*.py", "infer*.py", "run*.py", "eval*.py"):
        matches = sorted(model_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _generate_export_script(model_dir: Path, info: dict, checkpoint_path: Path,
                            output_path: Path, input_shape: str) -> str:
    """生成 ONNX 导出用的 Python 脚本内容（通用方案）"""
    model_name = info.get("model", "model")
    sub_model = info.get("sub_model", "")
    domain = info.get("domain", "")
    model_name_title = sub_model or model_name

    lines = [
        "#!/usr/bin/env python",
        "\"\"\"Auto-generated ONNX export script by onescience deploy\"\"\"",
        "import sys, os",
        "import importlib",
        "from pathlib import Path",
        "import torch",
        "",
        "# ---- 添加搜索路径 ----",
        f"sys.path.insert(0, {str(model_dir)!r})",
        f"sys.path.insert(0, {str(model_dir.parent)!r})",
        f"_models_path = {str((model_dir / 'models').resolve())!r}",
        "if os.path.isdir(_models_path) and _models_path not in sys.path:",
        "    sys.path.insert(0, _models_path)",
        "# onescience 源代码根路径（使 models.xxx 和 onescience.models.xxx 可导入）",
        f"_onescience_root = {str((model_dir.parent.parent.parent / 'src' / 'onescience').resolve())!r}",
        "if os.path.isdir(_onescience_root) and _onescience_root not in sys.path:",
        "    sys.path.insert(0, _onescience_root)",
        f"_src_root = {str((model_dir.parent.parent.parent / 'src').resolve())!r}",
        "if os.path.isdir(_src_root) and _src_root not in sys.path:",
        "    sys.path.insert(0, _src_root)",
        f"_project_root = {str(model_dir.parent.parent.parent.resolve())!r}",
        "if os.path.isdir(_project_root) and _project_root not in sys.path:",
        "    sys.path.insert(0, _project_root)",
        "",
        "# ---- 加载 checkpoint ----",
        f"ckpt = torch.load({str(checkpoint_path)!r}, map_location='cpu')",
        "",
        "# ---- 获取模型参数（优先级: checkpoint args > config 文件 > run.py parser） ----",
        "model_args = None",
        "if 'args' in ckpt:",
        "    model_args = ckpt['args']",
        "elif 'config' in ckpt:",
        "    model_args = ckpt['config']",
        "elif 'model_config' in ckpt:",
        "    model_args = ckpt['model_config']",
        "",
        "if model_args is None:",
        "    for _cfg_path in [",
        f"        {str(model_dir / 'conf' / 'config.yaml')!r},",
        f"        {str(model_dir / 'config.yaml')!r},",
        "    ]:",
        "        if os.path.exists(_cfg_path):",
        "            try:",
        "                import yaml",
        "                with open(_cfg_path) as f:",
        "                    _raw = yaml.safe_load(f)",
        "                    model_args = _raw.get('model', _raw)",
        "            except Exception:",
        "                pass",
        "            if model_args:",
        "                break",
        "",
        "# 从 run.py / train.py 的 argparse parser 获取默认参数（兜底）",
        "if model_args is None:",
        "    for _entry in ['run', 'train']:",
        "        try:",
        "            _entry_mod = __import__(_entry)",
        "            _parser = getattr(_entry_mod, 'parser', None)",
        "            if _parser is not None:",
        "                model_args = _parser.parse_args([])",
        "                break",
        "        except ImportError:",
        "            pass",
        "",
        "# ---- 导入模型类（多策略依次尝试） ----",
        f"model_cls_name = {model_name_title!r}",
        f"_domain = {domain!r}",
        f"_model_dir_name = {model_dir.name!r}",
        "ModelClass = None",
        "print(f'sys.path 前 5 项: {sys.path[:5]}')",
        "",
        "# 策略1: onescience.models.<pkg>.<ClassName>（完整路径避免相对导入问题）",
        f"for _pkg in ({domain.lower()!r}, {model_dir.name.lower()!r}):",
        "    _mod_path = f'onescience.models.{_pkg}.{model_cls_name}'",
        "    print(f'  策略1: 尝试 import {_mod_path} -> Model ...')",
        "    try:",
        "        _mod = importlib.import_module(_mod_path)",
        "        ModelClass = getattr(_mod, 'Model', None)",
        "        if ModelClass:",
        "            print(f'  策略1: 成功')",
        "            break",
        "        else:",
        "            print(f'  策略1: {_mod_path} 加载成功但无 class Model')",
        "    except Exception as _e:",
        "        print(f'  策略1: 失败 - {type(_e).__name__}: {_e}')",
        "",
        "# 策略2: onescience.models.<model_name>（如 onescience.models.pangu → Pangu）",
        "if ModelClass is None:",
        "    for _mn in [model_cls_name, model_cls_name.lower(), model_cls_name.lower().replace('_', '')]:",
        "        _mod_path2 = f'onescience.models.{_mn}'",
        "        print(f'  策略2: 尝试 import {_mod_path2} ...')",
        "        try:",
        "            _mod = importlib.import_module(_mod_path2)",
        "            ModelClass = getattr(_mod, 'Model', None)",
        "            if ModelClass is None:",
        "                for _attr in dir(_mod):",
        "                    if _attr.startswith('_'):",
        "                        continue",
        "                    _val = getattr(_mod, _attr, None)",
        "                    if isinstance(_val, type) and issubclass(_val, torch.nn.Module):",
        "                        ModelClass = _val",
        "                        break",
        "            if ModelClass:",
        "                print(f'  策略2: 成功 via {_mod_path2}')",
        "                break",
        "            else:",
        "                print(f'  策略2: 模块加载成功但无 nn.Module 子类')",
        "        except Exception as _e:",
        "            print(f'  策略2: 失败 - {type(_e).__name__}: {_e}')",
        "",
        "# 策略3: 以模型目录名作为模块名导入",
        f"if ModelClass is None:",
        f"    print(f'  策略3: 尝试 __import__(\"{model_dir.name}\") ...')",
        f"    try:",
        f"        _mod = __import__('{model_dir.name}', fromlist=['Model'])",
        f"        ModelClass = getattr(_mod, 'Model', None)",
        "        print(f'  策略3: ModelClass={ModelClass}')",
        "    except Exception as _e:",
        "        print(f'  策略3: 失败 - {type(_e).__name__}: {_e}')",
        "",
        "# 策略4: 从模型目录的入口脚本中提取模型导入",
        "if ModelClass is None:",
        "    import ast",
        "    print(f'  策略4: 扫描入口脚本寻找模型导入 ...')",
        f"    _entry_scripts = sorted(Path({str(model_dir)!r}).glob('train*.py')) + sorted(Path({str(model_dir)!r}).glob('infer*.py')) + sorted(Path({str(model_dir)!r}).glob('run*.py'))",
        "    print(f'  策略4: 找到脚本: {[s.name for s in _entry_scripts]}')",
        "    for _script in _entry_scripts[:3]:",
        "        try:",
        "            _tree = ast.parse(_script.read_text())",
        "            for _node in ast.walk(_tree):",
        "                if isinstance(_node, ast.ImportFrom) and _node.module and 'onescience.models' in _node.module:",
        "                    print(f'  策略4: 在 {_script.name} 中发现导入 {_node.module}.{_node.names}')",
        "                    for _alias in _node.names:",
        "                        _mod_name = _node.module.replace('onescience.models.', '', 1)",
        "                        try:",
        "                            _mod = importlib.import_module(f'onescience.models.{_mod_name}')",
        "                            _cls = getattr(_mod, _alias.name, None)",
        "                            if _cls and isinstance(_cls, type) and issubclass(_cls, torch.nn.Module):",
        "                                ModelClass = _cls",
        "                                print(f'  策略4: 成功 via onescience.models.{_mod_name}.{_alias.name}')",
        "                                break",
        "                        except Exception as _e2:",
        "                            print(f'  策略4: 导入 onescience.models.{_mod_name} 失败: {type(_e2).__name__}: {_e2}')",
        "                if ModelClass:",
        "                    break",
        "        except Exception as _e:",
        "            print(f'  策略4: 解析 {_script.name} 失败: {type(_e).__name__}: {_e}')",
        "        if ModelClass:",
        "            break",
        "    if ModelClass is None:",
        "        print(f'  策略4: 所有脚本均未找到有效模型导入')",
        "",
        "# 策略5: exec 动态导入（兜底）",
        "if ModelClass is None:",
        "    print(f'  策略5: 尝试 exec 导入 {model_cls_name} ...')",
        "    try:",
        f"        exec(f'from {model_name_title} import Model as ModelClass')",
        "        print(f'  策略5: 成功')",
        "    except Exception as _e:",
        "        print(f'  策略5: 失败 - {type(_e).__name__}: {_e}')",
        "",
        "if ModelClass is None:",
        "    print(f'错误: 无法自动找到模型类 {model_cls_name}')",
        f'    print(f"模型: {model_name_title}, 目录: {str(model_dir)!r}")',
        "    print('请手动编写导出脚本')",
        "    sys.exit(1)",
        "",
        "# ---- 实例化模型（多构造函数签名尝试） ----",
        "try:",
        "    if model_args is not None:",
        "        if isinstance(model_args, dict):",
        "            model = ModelClass(**model_args)",
        "        else:",
        "            try:",
        "                # 优先: (args, device) 风格 (CFD_Benchmark)",
        "                model = ModelClass(model_args, torch.device('cpu'))",
        "            except TypeError:",
        "                model = ModelClass(model_args)",
        "    else:",
        "        model = ModelClass()",
        "except TypeError:",
        "    # 参数不匹配时尝试无参实例化",
        "    try:",
        "        model = ModelClass()",
        "    except TypeError:",
        "        # 最后尝试带 device 的无参",
        "        model = ModelClass(None, torch.device('cpu'))",
        "",
        "# ---- 加载权重（按优先级尝试不同 key） ----",
        "for _key in ['model_state', 'model_state_dict', 'state_dict']:",
        "    if _key in ckpt:",
        "        model.load_state_dict(ckpt[_key], strict=False)",
        "        break",
        "else:",
        "    # 兜底: checkpoint 本身就是 state_dict",
        "    model.load_state_dict(ckpt, strict=False)",
        "",
        "model.eval()",
        "",
        "# ---- 导出 ONNX（通用方案：tuple 参数传递，无需模块包装） ----",
        f"shape = tuple(int(x) for x in '{input_shape}'.split(','))",
        "dummy_input = torch.randn(shape)",
        f"_output_path = {str(output_path)!r}",
        "print(f'导出 ONNX 模型到: {_output_path}')",
        "",
        "import inspect as _inspect",
        "sig = _inspect.signature(model.forward)",
        "_param_list = [p for p in sig.parameters.values() if p.name != 'self']",
        "_export_kw = dict(opset_version=15, input_names=['input'], output_names=['output'],",
        "                 dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})",
        "",
        "def _try_export(_args, _desc):",
        "    try:",
        "        torch.onnx.export(model, _args, _output_path, **_export_kw)",
        "        print(f'  导出成功: {_desc}')",
        "        return True",
        "    except Exception as _e:",
        "        print(f'    {_desc}: {type(_e).__name__}: {_e}')",
        "        return False",
        "",
        "# 策略A: 单输入模型",
        "if len(_param_list) == 1:",
        "    if _try_export(dummy_input, '单输入'): exit(0)",
        "    else: print(f'单输入导出失败'); exit(1)",
        "",
        "# ---- 从 model.args 推断输入维度（通用） ----",
        "_space_dim = None",
        "_fun_dim = None",
        "if hasattr(model, 'args'):",
        "    _space_dim = getattr(model.args, 'space_dim', None)",
        "    _fun_dim = getattr(model.args, 'fun_dim', None)",
        "    # 有些模型用 in_dim 或 input_dim",
        "    if _space_dim is None:",
        "        _space_dim = getattr(model.args, 'in_dim', None)",
        "    if _space_dim is None:",
        "        _space_dim = getattr(model.args, 'input_dim', None)",
        "",
        "# 如果用户输入 shape 与模型期望不匹配，自动 reshape",
        "_x = dummy_input",
        "# 优先使用 shapelist 创建正确形状的输入（CFD 等网格模型）",
        "_shapelist = getattr(model.args, 'shapelist', None) if hasattr(model, 'args') else None",
        "if _shapelist is not None and len(_shapelist) > 0 and _space_dim is not None:",
        "    _N = 1",
        "    for _s in _shapelist:",
        "        _N *= _s",
        "    _x = torch.randn(dummy_input.size(0), _N, _space_dim)",
        "    print(f'  使用 shapelist({_shapelist}) 创建输入: {_x.shape}')",
        "elif _space_dim is not None and _space_dim <= 3 and dummy_input.dim() == 4:",
        "    _total = dummy_input.size(1) * dummy_input.size(2) * dummy_input.size(3)",
        "    if _total % _space_dim == 0:",
        "        _x = dummy_input.reshape(dummy_input.size(0), -1, _space_dim)",
        "        print(f'  输入 reshape: {dummy_input.shape} -> {_x.shape}')",
        "    else:",
        "        _x = torch.randn(dummy_input.size(0), _total, _space_dim)",
        "        print(f'  输入 reshape (重建): {dummy_input.shape} -> {_x.shape}')",
        "",
        "# 多输入模型：动态构建 args tuple",
        "_param_names = [p.name for p in _param_list]",
        "print(f'  模型参数: {_param_names}')",
        "",
        "# 策略B: None/default 填充额外参数",
        "_argsB = [_x]",
        "for _p in _param_list[1:]:",
        "    if _p.default is not _inspect.Parameter.empty:",
        "        _argsB.append(_p.default)",
        "    else:",
        "        # 对 fx/coord 类参数：fun_dim==0 时应传 None",
        "        if _p.name in ('fx', 'coord', 'coords') and _fun_dim is not None and _fun_dim == 0:",
        "            _argsB.append(None)",
        "        elif _p.name == 'geo':",
        "            _argsB.append(None)",
        "        else:",
        "            _argsB.append(None)",
        "if _try_export(tuple(_argsB), '多输入(None填充)'):",
        "    exit(0)",
        "",
        "# 策略C: 从 model.args 推断形状并 reshape",
        "if hasattr(model, 'args') and _space_dim is not None:",
        "    _argsC = [_x]",
        "    for _p in _param_list[1:]:",
        "        if _p.name in ('fx', 'coord', 'coords'):",
        "            if _fun_dim is not None and _fun_dim > 0:",
        "                _argsC.append(torch.zeros(_x.size(0), _x.size(1), _fun_dim))",
        "            else:",
        "                _argsC.append(None)",
        "        elif _p.name in ('T', 'time', 't'):",
        "            _argsC.append(torch.zeros(_x.size(0), 1))",
        "        elif _p.name == 'geo':",
        "            _argsC.append(None)",
        "        else:",
        "            _argsC.append(torch.zeros(_x.size(0), 1))",
        "    if _try_export(tuple(_argsC), '多输入(args推断)'):",
        "        exit(0)",
        "",
        "# 策略D: 全零张量兜底（batch 对齐）",
        "_argsD = [_x]",
        "for _p in _param_list[1:]:",
        "    if _p.name in ('fx', 'coord', 'coords'):",
        "        _argsD.append(torch.zeros(_x.size(0), 2))",
        "    elif _p.name == 'geo':",
        "        _argsD.append(None)",
        "    else:",
        "        _argsD.append(torch.zeros(1))",
        "if _try_export(tuple(_argsD), '多输入(全零)'):",
        "    exit(0)",
        "",
        "# 全部失败，打印诊断",
        "print(f'错误: 所有策略均失败')",
        "print(f'模型 forward 签名: {sig}')",
        "print(f'模型 forward 参数: {_param_names}')",
        "print(f'用户指定输入形状: {shape}')",
        "exit(1)",
        "",
        "print('导出完成')",
        "",
    ]

    return "\n".join(lines)


@deploy_group.command("export")
@click.argument("model_alias")
@click.option("--input-shape", "-s", default="1,3,224,224",
              help="模型输入形状，逗号分隔，如 1,3,224,224")
@click.option("--output", "-o", default=None,
              help="ONNX 文件输出路径（默认: model_dir/model.onnx）")
@click.option("--checkpoint", "-c", default=None,
              help="指定 checkpoint 文件路径（默认自动查找最新的）")
@click.option("--force", "-f", is_flag=True, default=False,
              help="覆盖已有 ONNX 文件")
@click.option("--dry-run", is_flag=True, default=False,
              help="仅打印导出脚本内容，不实际执行")
def export_model(model_alias, input_shape, output, checkpoint, force, dry_run):
    """导出训练好的 PyTorch 模型为 ONNX 格式"""
    info = model_registry.resolve(model_alias, download=False)
    if not info:
        click.secho(f"未知模型: {model_alias}", fg="red")
        raise SystemExit(1)

    model_dir = get_model_dir(info)
    if not model_dir or not model_dir.exists():
        click.secho(f"模型目录不存在: {model_dir}", fg="red")
        raise SystemExit(1)

    sub_model = info.get("sub_model", "")
    model_name = sub_model or info.get("model", model_alias)

    # 定位 checkpoint
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            click.secho(f"checkpoint 文件不存在: {ckpt_path}", fg="red")
            raise SystemExit(1)
    else:
        checkpoints = _find_checkpoints(model_dir)
        if not checkpoints:
            click.secho(
                f"未在 {model_dir} 下找到 checkpoint 文件 (*.pth / *.pt)",
                fg="red",
            )
            click.echo("请先训练模型，或使用 -c 指定 checkpoint 路径")
            raise SystemExit(1)
        # 按修改时间取最新的
        ckpt_path = max(checkpoints, key=lambda p: p.stat().st_mtime)

    # 确定输出路径
    if output:
        output_path = Path(output)
    else:
        output_path = model_dir / f"{model_name}.onnx"

    if output_path.exists() and not force:
        click.secho(f"ONNX 文件已存在: {output_path}", fg="yellow")
        click.echo("使用 --force / -f 覆盖")
        raise SystemExit(1)

    # 显示导出信息
    click.secho(f"模型导出: {model_alias}", fg="green")
    click.echo(f"  模型:      {info.get('model', '?')} ({info.get('domain', '?')})")
    if sub_model:
        click.echo(f"  子模型:    {sub_model}")
    click.echo(f"  目录:      {model_dir}")
    click.echo(f"  Checkpoint: {ckpt_path}")
    click.echo(f"  输出:      {output_path}")
    click.echo(f"  输入形状:  {input_shape}")
    click.echo("")

    # 生成导出脚本
    script_content = _generate_export_script(
        model_dir, info, ckpt_path, output_path, input_shape,
    )

    if dry_run:
        click.secho("导出脚本预览（--dry-run，未执行）:", fg="cyan")
        click.echo("")
        click.echo(script_content)
        return

    # 写入临时脚本并执行
    from ..core.runner import _run_cmd
    import tempfile

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    )
    try:
        tmp.write(script_content)
        tmp.close()

        # 构建执行环境
        import os
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        click.secho("正在导出 ONNX 模型...", fg="cyan")
        log_path = model_dir / f"{model_name}_export_onnx.log"

        result = _run_cmd(
            cmd=[sys.executable, tmp.name],
            cwd=model_dir,
            log_path=log_path,
            env=env,
        )
    finally:
        os.unlink(tmp.name)

    if result and result.get("success"):
        click.secho(f"导出成功: {output_path}", fg="green")
        click.echo(f"  日志: {log_path}")
    else:
        click.secho(f"导出失败，请查看日志: {log_path}", fg="red")
        raise SystemExit(1)


@deploy_group.command("serve")
@click.argument("model_alias")
@click.option("--port", "-p", default=8000, type=int, help="HTTP 服务端口")
@click.option("--backend", "-b", default="triton", type=click.Choice(["triton"]),
              help="推理后端")
@click.option("--model-name", "-n", default=None,
              help="Triton 模型仓库中的模型名（默认: 模型别名）")
@click.option("--model-repo", "-r", default=None,
              help="Triton 模型仓库目录（默认: model_dir/triton_repo）")
@click.option("--gpu", is_flag=True, default=True,
              help="使用 GPU 推理")
@click.option("--dry-run", is_flag=True, default=False,
              help="仅打印启动命令，不实际启动服务")
def serve_model(model_alias, port, backend, model_name, model_repo, gpu, dry_run):
    """启动模型推理服务（Triton）"""
    info = model_registry.resolve(model_alias, download=False)
    if not info:
        click.secho(f"未知模型: {model_alias}", fg="red")
        raise SystemExit(1)

    model_dir = get_model_dir(info)
    if not model_dir or not model_dir.exists():
        click.secho(f"模型目录不存在: {model_dir}", fg="red")
        raise SystemExit(1)

    # 确定 ONNX 文件路径
    sub_model = info.get("sub_model", "")
    model_name_tag = model_name or sub_model or info.get("model", model_alias)
    onnx_path = model_dir / f"{model_name_tag}.onnx"

    if not onnx_path.exists():
        click.secho(f"未找到 ONNX 模型文件: {onnx_path}", fg="red")
        click.echo("请先运行 'onescience deploy export' 导出模型")
        raise SystemExit(1)

    # 设置 Triton 模型仓库
    if model_repo:
        repo_dir = Path(model_repo)
    else:
        repo_dir = model_dir / "triton_repo"

    # Triton 模型仓库结构:
    #   triton_repo/
    #     <model_name>/
    #       1/
    #         model.onnx
    #       config.pbtxt

    model_repo_path = repo_dir / model_name_tag / "1"
    model_repo_path.mkdir(parents=True, exist_ok=True)

    # 复制 ONNX 文件
    target_onnx = model_repo_path / "model.onnx"
    if not target_onnx.exists():
        shutil.copy2(str(onnx_path), str(target_onnx))

    # 生成 config.pbtxt
    config_pbtxt = repo_dir / model_name_tag / "config.pbtxt"
    if not config_pbtxt.exists():
        gpu_str = "true" if gpu else "false"
        config_content = f"""name: "{model_name_tag}"
backend: "onnxruntime"
max_batch_size: 8
input {{
  name: "input"
  data_type: TYPE_FP32
  dims: [ 3, 224, 224 ]
}}
output {{
  name: "output"
  data_type: TYPE_FP32
  dims: [ 3, 224, 224 ]
}}
instance_group {{
  count: 1
  kind: KIND_GPU
}}
"""
        config_pbtxt.write_text(config_content, encoding="utf-8")

    click.secho(f"推理服务: {model_alias}", fg="green")
    click.echo(f"  模型:      {info.get('model', '?')} ({info.get('domain', '?')})")
    click.echo(f"  后端:      {backend}")
    click.echo(f"  端口:      {port}")
    click.echo(f"  ONNX:      {onnx_path}")
    click.echo(f"  仓库:      {repo_dir}")
    click.echo(f"  GPU:       {gpu}")
    click.echo("")

    # Triton 启动命令
    triton_cmd = (
        f"tritonserver --model-repository={repo_dir} "
        f"--http-port={port} "
        f"--grpc-port={port + 1} "
        f"--metrics-port={port + 2}"
    )

    if dry_run:
        click.secho("启动命令（--dry-run，未执行）:", fg="cyan")
        click.echo("")
        click.echo(triton_cmd)
        return

    # 检查 tritonserver 是否可用
    if not shutil.which("tritonserver"):
        click.secho("未找到 tritonserver 命令", fg="red")
        click.echo("请先安装 Triton Inference Server")
        click.echo("或使用 --dry-run 查看手动启动方式")
        click.echo("")
        click.echo(triton_cmd)
        raise SystemExit(1)

    click.secho("正在启动 Triton 推理服务...", fg="cyan")
    click.echo(f"  命令: {triton_cmd}")
    click.echo("  按 Ctrl+C 停止服务")

    try:
        subprocess.run(
            triton_cmd.split(),
            cwd=str(model_dir),
        )
    except KeyboardInterrupt:
        click.echo("")
        click.secho("服务已停止", fg="yellow")
    except FileNotFoundError:
        click.secho("未找到 tritonserver 命令", fg="red")
        raise SystemExit(1)
