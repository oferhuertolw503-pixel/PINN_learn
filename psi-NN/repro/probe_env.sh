#!/usr/bin/env bash
# 环境探针：只读地打印 WSL2 Ubuntu 里与本次复现相关的全部事实。
# 用法（Windows 侧）：
#   wsl.exe -d Ubuntu-24.04 -- bash /mnt/d/work/research/PINN/psi-NN/repro/probe_env.sh

echo "===== 1. 系统 ====="
echo "user  : $(whoami)   home=$HOME"
. /etc/os-release && echo "distro: $PRETTY_NAME"
echo "kernel: $(uname -r)   arch: $(uname -m)"

echo
echo "===== 2. WSL 互操作 / GPU 设备 ====="
echo "WSL_DISTRO_NAME=${WSL_DISTRO_NAME:-<unset>}"
ls -l /dev/dxg 2>&1 | head -2
echo "nvidia-smi 路径: $(command -v nvidia-smi || echo '<不在 PATH>')"
[ -x /usr/lib/wsl/lib/nvidia-smi ] && /usr/lib/wsl/lib/nvidia-smi \
    --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | head -3

echo
echo "===== 3. Python / pip / venv 状态（关键） ====="
echo "python3 : $(command -v python3 || echo '<NOT FOUND>')  $(python3 --version 2>&1)"
echo "pip     : $(python3 -m pip --version 2>&1 | head -1)"
python3 - <<'PY'
for mod in ('venv', 'ensurepip'):
    try:
        __import__(mod)
        print(f"{mod:10}: OK")
    except Exception as e:
        print(f"{mod:10}: MISSING -> {e}")
PY
echo "-- apt 包状态 --"
dpkg-query -W -f='${Package} | ${Status} | ${Version}\n' \
    python3-venv python3.12-venv python3-pip python3-dev 2>&1

echo
echo "===== 4. sudo 可用性（装 apt 包要用） ====="
if sudo -n true 2>/dev/null; then
    echo "免密 sudo : 可用"
else
    echo "免密 sudo : 不可用（装 apt 包时会要求输密码）"
fi
echo "apt-get   : $(command -v apt-get || echo '<NOT FOUND>')"

echo
echo "===== 5. libcuda 对动态链接器可见性（torch 找 GPU 靠它） ====="
echo "ldconfig 命中 libcuda: $(ldconfig -p 2>/dev/null | grep -c libcuda)"
ls /etc/ld.so.conf.d/ 2>/dev/null | grep -i wsl || echo "(没有 wsl 相关的 ld.so.conf.d 条目)"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

echo
echo "===== 6. 网络：pip 源可达性（测真实包索引页，不是 /simple/ 根） ====="
for m in https://pypi.org/simple \
         https://pypi.tuna.tsinghua.edu.cn/simple \
         https://mirrors.aliyun.com/pypi/simple \
         https://mirrors.ustc.edu.cn/pypi/simple \
         https://repo.huaweicloud.com/repository/pypi/simple ; do
    printf '%-58s ' "$m"
    curl -sS -o /dev/null -m 15 -w 'HTTP %{http_code}  %{time_total}s  %{speed_download} B/s\n' \
        "$m/pip/" 2>&1 || echo "不可达"
done
echo "-- torch 专用源 --"
printf '%-58s ' https://download.pytorch.org/whl/cu128/
curl -sS -o /dev/null -m 15 -w 'HTTP %{http_code}  %{time_total}s\n' https://download.pytorch.org/whl/cu128/ 2>&1
echo "-- apt 源配置与可达性 --"
grep -rhE '^(deb|URIs:)' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null | head -6
apt_mirror=$(grep -rhoE 'https?://[^ ]+' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null | head -1)
[ -n "$apt_mirror" ] && { printf '%-58s ' "$apt_mirror"; curl -sS -o /dev/null -m 15 -w 'HTTP %{http_code}  %{time_total}s\n' "$apt_mirror" 2>&1; }

echo
echo "===== 7. 已有 Python 包 ====="
python3 -m pip list 2>/dev/null | grep -iE 'torch|numpy|pandas|matplotlib|scipy' || echo "(pip 不可用或未装相关包)"

echo
echo "===== 8. 编译工具链 ====="
for t in gcc g++ make git curl; do printf '%-5s: %s\n' "$t" "$(command -v $t || echo '<NOT FOUND>')"; done

echo
echo "===== 9. 磁盘 / 挂载 ====="
df -h / /mnt/d 2>&1 | cat
mount | grep ' /mnt/d ' | sed 's/,/,\n    /g' || true
echo "-- 已有 venv --"
ls -d "$HOME"/.venvs/* 2>/dev/null || echo "(还没有 \$HOME/.venvs)"

echo
echo "===== 10. 绘图字体 ====="
echo "Times New Roman: $(fc-list 2>/dev/null | grep -ic 'times new roman') 个"
echo "DejaVu Sans    : $(fc-list 2>/dev/null | grep -ic 'dejavu sans') 个"

echo
echo "===== 11. 目标目录可写性 ====="
REPO=/mnt/d/work/research/PINN/psi-NN/Psi-NN
[ -d "$REPO" ] && echo "代码目录: $REPO  存在" || echo "!! 代码目录不存在: $REPO"
if mkdir -p "$REPO/Results/_probe" 2>/dev/null; then
    echo "创建 Results/ 并写文件: OK"
    rm -rf "$REPO/Results/_probe"
else
    echo "!! 无法创建 $REPO/Results/"
fi
