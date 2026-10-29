# Mac 学习环境：uv + .venv

2026-10-29 环境迁移：在新仓库根目录重新建立 `.venv`，使用 uv 管理的 Python 3.12.14，依赖版本记录在 `requirements-mac-learning.txt`。不再依赖 Conda、旧项目目录或 Codex 自带的 Python；WSL CUDA 实验环境保持原样。

## 日常使用

在新仓库根目录打开终端，只需：

```bash
source .venv/bin/activate
python --version
```

激活只对当前终端生效。提示符出现 `(.venv)` 后，`python` 使用这个项目的解释器；也可以一直写 `.venv/bin/python` 而不激活。退出环境用 `deactivate`。

短程检查：

```bash
python shred/repro/smoke_original.py --epochs 2 --device cpu
python psi-NN/repro/run_mac.py --case Laplace --device cpu --steps 2 --models PINN --tag env-check
```

这些只检查运行链路，不验证论文精度。结果进入被忽略的 Results，不覆盖 Git 里的正式实验摘要。

## 需要重新创建时

已有 uv、环境目录尚不存在时，可在仓库根目录按记录的版本复建：

```bash
uv python install 3.12.14
uv venv --python 3.12.14 --managed-python .venv
uv pip install --python .venv/bin/python -r requirements-mac-learning.txt
uv pip check --python .venv/bin/python
```

Python 是解释器；`.venv` 是项目自己的环境；torch/numpy 等是装在其中的依赖。uv 管理独立解释器与包安装，所以不需要把已删除的 Conda 装回来。

不要直接搬运旧 `.venv`：启动脚本和解释器配置可能包含旧绝对路径。源码、论文和数据可以复制；环境按版本记录在目标位置重建。[Python 官方说明](https://docs.python.org/3.12/library/venv.html)、[uv 环境说明](https://docs.astral.sh/uv/pip/environments/)。

`.venv` 不提交 Git；依赖清单和使用说明可以同步。清单仅用于 Mac 阅读/短程检查，不能拿它覆盖实验机的 CUDA PyTorch，也不代表已安装 PySHRED 软件包或 h5py。PySHRED 的依赖要求与当前 NumPy 版本不同，后续使用时另行处理。
