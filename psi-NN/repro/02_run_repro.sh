#!/usr/bin/env bash
# =============================================================================
# Psi-NN 最小复现 —— 步骤 2：Burgers 逆问题「蒸馏 + 结构提取」一条龙
#
# 用法（在 WSL 的 Ubuntu 终端里）：
#     bash psi-NN/repro/02_run_repro.sh
#
# 可选环境变量：
#     STEPS=30000    覆盖训练步数（默认用 Config/Burgers_inv_distill_REPRO.csv 里的 10000）
#     QUES / INI     换算例（默认 Burgers_inv_distill REPRO）
#     REPO_DIR=...   代码根目录
#     VENV_DIR=...   虚拟环境目录
# =============================================================================
set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$REPRO_DIR/../Psi-NN}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/psinn}"
QUES="${QUES:-Burgers_inv_distill}"
INI="${INI:-REPRO}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "找不到虚拟环境：$VENV_DIR"
    echo "请先准备已有依赖的环境；不要未经确认运行安装脚本。"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [ ! -d "$REPO_DIR/Module" ]; then
    echo "找不到作者代码：$REPO_DIR/Module"
    echo "请在 psi-NN/Psi-NN 检出作者源码，或设置 REPO_DIR。"
    exit 1
fi
cd "$REPO_DIR"
LOG_DIR="$REPO_DIR/Results/_repro_logs"
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/${QUES}_${INI}_train.log"
EXTRACT_LOG="$LOG_DIR/${QUES}_${INI}_extract.log"

if [ -n "${STEPS:-}" ]; then
    echo "覆盖训练步数 -> $STEPS"
    sed -i -E "s/^train_steps,.*/train_steps, ${STEPS}, 单组训练步数, train steps per group/" \
        "Config/${QUES}_${INI}.csv"
fi

echo
echo "############ 1/2 训练：教师 PINN + 学生蒸馏 ############"
python -u "$REPRO_DIR/run_train.py" "$QUES" "$INI" 2>&1 | tee "$TRAIN_LOG"

echo
echo "############ 2/2 结构提取：权重聚类 -> 字母矩阵 ############"
python -u "$REPRO_DIR/extract_structure.py" --ques "$QUES" --ini "$INI" 2>&1 | tee "$EXTRACT_LOG"

echo
echo "############ 完成 ############"
echo "结果目录 : $REPO_DIR/Results/${QUES}_${INI}/"
echo "训练日志 : $TRAIN_LOG"
echo "提取日志 : $EXTRACT_LOG"
