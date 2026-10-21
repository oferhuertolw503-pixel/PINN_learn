#!/usr/bin/env bash
# =============================================================================
# Psi-NN 最小复现 —— 步骤 1：在 WSL2 Ubuntu 里准备 Python 环境
#
# 用法（在 WSL 的 Ubuntu 终端里）：
#     bash /mnt/d/work/research/PINN/psi-NN/repro/01_setup_env.sh
#
# 脚本针对本机实测到的三个坑做了处理：
#   1. Ubuntu 24.04 默认没装 python3-venv / python3-pip。此时 `python3 -m venv`
#      会**成功返回 0**，但生成的是一个没有 pip 的 venv，后面 `pip install` 直接
#      报 "No module named pip"。所以这里先检测 ensurepip / pip 并用 apt 补齐。
#   2. 本机 WSL 里 pypi.org 不可达（OpenSSL SSL_ERROR_SYSCALL），必须换国内镜像。
#      脚本按实测顺序探测 阿里云 / 华为云 / USTC / 清华，选第一个可用的。
#   3. RTX 5060 Laptop 是 Blackwell(sm_120)，必须 cu128 及以上轮子，否则报
#      "no kernel image is available"。torch 单独走 download.pytorch.org
#      （本机实测 1.5s 可达），其余包走上面选出的镜像。
#
# 可用环境变量覆盖：VENV_DIR / PIP_INDEX / TORCH_INDEX
# =============================================================================
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/.venvs/psinn}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "=============================================="
echo " Psi-NN 复现环境安装"
echo "=============================================="
echo "发行版  : $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "内核    : $(uname -r)"
echo "venv    : $VENV_DIR"
echo

# ---------------------------------------------------------------- 1. GPU 透传
echo "== [1/5] GPU 透传检查 =="
if [ -e /dev/dxg ]; then
    echo "/dev/dxg 存在（WSL GPU 透传正常）"
else
    echo "!! /dev/dxg 不存在，WSL 可能没做 GPU 透传；先在 Windows 侧执行 wsl --update"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "!! 找不到 nvidia-smi（正常应位于 /usr/lib/wsl/lib/ 并已在 PATH 中）"
fi
echo

# ------------------------------------------------------- 2. 预检 pip / venv
echo "== [2/5] 预检 pip / ensurepip =="
APT_PKGS=()
python3 -c 'import ensurepip' 2>/dev/null || APT_PKGS+=(python3.12-venv)
python3 -m pip --version >/dev/null 2>&1 || APT_PKGS+=(python3-pip)

if [ ${#APT_PKGS[@]} -gt 0 ]; then
    echo "缺少: ${APT_PKGS[*]}"
    if sudo -n true 2>/dev/null; then
        echo "--> apt 安装中（免密 sudo 可用；本机 apt 源已是 mirrors.aliyun.com）"
        sudo apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 \
            -o Acquire::https::Timeout=30 -o Acquire::Languages=none update
        sudo apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 \
            -o Acquire::https::Timeout=30 install -y "${APT_PKGS[@]}"
    else
        echo "!! 需要你自己执行（会要求输入密码）："
        echo "     sudo apt-get update && sudo apt-get install -y ${APT_PKGS[*]}"
        exit 1
    fi
else
    echo "pip 与 ensurepip 均已就绪"
fi
python3 -c 'import ensurepip' && echo "ensurepip: OK"
echo

# ------------------------------------------------------------ 3. 选 pip 镜像
echo "== [3/5] 选择 pip 镜像 =="
# 本机代理对国内镜像和 PyPI 文件 CDN 的 TLS 连接超时；这些直连，PyTorch 源沿用代理。
export no_proxy="${no_proxy:-${NO_PROXY:-}},mirrors.aliyun.com,repo.huaweicloud.com,mirrors.ustc.edu.cn,pypi.tuna.tsinghua.edu.cn,files.pythonhosted.org"
export NO_PROXY="$no_proxy"
if [ -n "${PIP_INDEX:-}" ]; then
    echo "使用外部指定: $PIP_INDEX"
else
    PIP_INDEX=""
    for m in https://mirrors.aliyun.com/pypi/simple \
             https://repo.huaweicloud.com/repository/pypi/simple \
             https://mirrors.ustc.edu.cn/pypi/simple \
             https://pypi.tuna.tsinghua.edu.cn/simple \
             https://pypi.org/simple ; do
        printf '  探测 %-56s ' "$m"
        if curl -fsS -o /dev/null -m 10 "$m/pip/" 2>/dev/null; then
            echo "可用"
            PIP_INDEX="$m"
            break
        fi
        echo "不可达"
    done
    if [ -z "$PIP_INDEX" ]; then
        echo "!! 没有可用的 pip 源，请用 PIP_INDEX=https://... 手动指定"
        exit 1
    fi
fi
echo "选定: $PIP_INDEX"
echo

# ------------------------------------------------------ 4. 建 venv 装依赖
echo "== [4/5] 创建 venv 并安装依赖 =="
if [ -d "$VENV_DIR" ]; then
    echo "已存在，跳过创建：$VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo "已创建：$VENV_DIR"
fi

# 把这个 venv 的默认源写死，之后手动 pip install 也会走镜像
cat > "$VENV_DIR/pip.conf" <<EOF
[global]
index-url = $PIP_INDEX
EOF

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "-- PyTorch（CUDA 12.8 轮子；Blackwell sm_120 必需）--"
# 注意这里是 --index-url 而不是 --extra-index-url：只从 pytorch 源取，
# 避免从镜像里挑到一个 CPU 版 torch 而“静默装错”。
pip install torch --index-url "$TORCH_INDEX"

echo "-- 其余依赖 --"
pip install numpy pandas matplotlib scipy
echo

# ------------------------------------------------------------------ 5. 自检
echo "== [5/5] 自检 =="
python - <<'PY'
import sys
import torch, numpy, pandas, matplotlib, scipy

print(f"python      : {sys.version.split()[0]}")
print(f"torch       : {torch.__version__}")
print(f"numpy       : {numpy.__version__}")
print(f"pandas      : {pandas.__version__}")
print(f"scipy       : {scipy.__version__}")
print(f"torch cuda  : {torch.version.cuda}")
print(f"cuda avail  : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print()
    print("!! CUDA 不可用：Training.py 会自动退到 CPU 训练，本复现会慢到不可接受。")
    print("   常见原因：装成了 CPU 版 torch / Windows 侧驱动太旧 / WSL 没做 GPU 透传。")
    sys.exit(1)

print(f"device      : {torch.cuda.get_device_name(0)}")
print(f"capability  : sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")

a = torch.randn(2048, 2048, device='cuda')
b = torch.randn(2048, 2048, device='cuda')
for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
print(f"matmul      : OK (checksum {float(c.sum()):.4f})")

# Training.py 的 net_f() 真正依赖的算子：对输入坐标求二阶导 u_xx / u_yy
x = torch.randn(64, 1, device='cuda', requires_grad=True)
u = torch.tanh(2 * x)
u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                          retain_graph=True, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                           retain_graph=True, create_graph=True)[0]
assert u_x.shape == x.shape and u_xx.shape == x.shape
print("autograd 2nd: OK (u_xx / u_yy 可用)")

print()
print("环境就绪。下一步：bash 02_run_repro.sh")
PY
