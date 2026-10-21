# Ψ-NN 个人复现脚本

> **公开仓库入口**：请先读 [`../../README.md`](../../README.md) 和 [`../../docs/phase-1-psi-nn.md`](../../docs/phase-1-psi-nn.md)。本页主要保留 Burgers 早期操作记录；其中的绝对路径、环境安装命令和“Laplace 未运行”等表述属于 2026-10-15 以前的历史快照，不是跨机器执行清单。当前已完成的 Laplace / Burgers 数值和方法边界以第一阶段记录为准。运行环境以现有依赖为前提，任何新安装需另行确认。

论文：*Automatic network structure discovery of physics informed neural networks via knowledge distillation*, Nature Communications 16, 9558 (2025).

截至 2026-10-15，已完成论文配置的教师 100,000 步、学生 200,000 步，以及 PINN、PINN-post、作者仓库手写 `PsiNN_burgers` 各 100,000 步的比较实验。也完成了 average/Ward 两种聚类和一个 37 参数的受约束重建试验。手写结构的预测误差低于两个基线，但**从学生权重自动提取的结构没有达到论文最终 Ψ-NN 的效果**；因此这是部分复现，不能称为完整方法复现。

> 当前结果和误差口径以[公开的第一阶段记录](../../docs/phase-1-psi-nn.md)为准。本页下文保留早期低步数试跑的操作说明；其中环境安装和“尚未跑全量”等旧表述不代表当前状态。实验机已有环境，Mac 阅读机无需安装训练依赖。

---

## 1. 复现了什么 / 没复现什么

**已执行**

| 论文步骤 | 代码落点 | 本次是否执行 |
|---|---|---|
| ① 教师 PINN 训练 | `Module/Training.py` → `train_adam()` 前半段 | ✅ |
| ② 知识蒸馏得到学生网络 | `net_teach()` + `net_d(mode='student')` | ✅ |
| ③ 从学生权重里提取符号化结构 | `pic_parameter.ipynb` 第 2、4 单元 | ✅（已做成无头脚本） |
| ④ 以固定聚类关系重训 | `repro/clustered_burgers.py` / `run_reconstructed.py` | ✅ 37 参数试验；未复现作者完整的块合并与重初始化 |
| ⑤ 比较模型 | `Module/PsiNN_burgers.py`、PINN、PINN-post | ✅ 各 100,000 步；`PsiNN_burgers` 是预写结构，不是步骤③的自动产物 |

**截至该 Burgers 阶段未做的**：论文完整的自动结构重构、Laplace / Poisson / Flow 三个算例，以及跨控制参数迁移实验。此后 Laplace 已完成部分复现，见公开的第一阶段记录。

---

## 2. 前置条件

- WSL2 + Ubuntu 24.04（本机已装）
- Windows 侧 NVIDIA 驱动 ≥ 570（本机 595.79）—— WSL 里的 CUDA 直接用这个驱动，**不要在 WSL 里装显卡驱动**
- 代码根目录：`/mnt/d/work/research/PINN/psi-NN/Psi-NN`

## 3. 三步操作

### 步骤 1：装环境（一次性）

```bash
bash /mnt/d/work/research/PINN/psi-NN/repro/01_setup_env.sh
```

- 虚拟环境建在 `~/.venvs/psinn`（**故意放 ext4 家目录**，不放 `/mnt/d`：drvfs 装包又慢、软链接还容易坏）
- 装 `torch` 的 **CUDA 12.8** 轮子 —— RTX 5060 是 Blackwell（sm_120），装默认的旧版 torch 会报
  `no kernel image is available for execution on the device`
- 最后自检 GPU，并单独验证 PINN 真正依赖的算子（对输入坐标求二阶导）。看到 `环境就绪` 才算过。

脚本会自动处理本机实测到的两个**阻塞项**：

- **没装 `python3-venv` / `python3-pip`，`ensurepip` 也缺失**。此时 `python3 -m venv` 会「成功」返回 0，但建出来的是一个**没有 pip** 的 venv，后面 `pip install` 直接报 `No module named pip`。脚本会先探测并用 `sudo apt-get` 补齐（本机免密 sudo 可用）。
- **WSL 里 `pypi.org` 不可达**（`OpenSSL SSL_connect: SSL_ERROR_SYSCALL`）。脚本按实测延迟依次探测 阿里云 / 华为云 / USTC / 清华，选第一个可用的，并把选中的源写进 venv 的 `pip.conf`。torch 单独走 `download.pytorch.org/whl/cu128`（实测 1.5 s 可达），且用 `--index-url` 而非 `--extra-index-url`，避免从镜像里挑到 CPU 版 torch 而静默装错。

torch 那步下载量较大（约 2–3 GB，含 cudnn/cublas 等 nvidia 依赖），是整条流程里最慢的一步。

### 步骤 2：跑复现

```bash
bash /mnt/d/work/research/PINN/psi-NN/repro/02_run_repro.sh
```

默认教师 10000 步 + 学生 20000 步（`train_ratio=2`）。想更接近论文就加大步数：

```bash
STEPS=50000 bash /mnt/d/work/research/PINN/psi-NN/repro/02_run_repro.sh
```

### 步骤 3：看产物

全部落在 `Psi-NN/Results/Burgers_inv_distill_REPRO/`：

```
Models/
  Burgers_inv_distill_REPRO_PINN.pth                        最终教师权重
  Burgers_inv_distill_REPRO_PINN_student.pth                最终学生权重 ← 结构提取就是读它
  Burgers_inv_distill_REPRO_PINN_step_*.pth                 教师中间快照
  Burgers_inv_distill_REPRO_PINN_student_step_*.pth         学生中间快照
Loss/
  Burgers_inv_distill_REPRO_loss_PINN.csv                   教师 loss 曲线数据
  Burgers_inv_distill_REPRO_loss_PINN_student.csv           学生 loss 曲线数据
Parameters/
  ..._paras_PINN.csv                                        反演出的 λ 随步数的变化
Figure/                                                     场分布图
Cluster/
  ..._fc1.weight_distribution.png                           逐层权重分布 + 聚类中心
  Burgers_inv_distill_REPRO_structure.json                  ★ 结构化结果（字母矩阵，机器可读）
Clock time.csv                                              纯训练耗时（不含画图）
```

`Cluster/*.json` 是这次最关键的产物：每层的 `symbols`（A/B/-A/-B 矩阵）和 `centers`（簇中心的数值）。

---

## 4. 怎么判断复现成功

1. **教师收敛**：`Loss_f` 从 ~1e0 降到 1e-3 量级；`Parameters/` 里反演出的 λ 往真值 `0.01/π ≈ 0.00318` 靠。
   （Burgers 逆问题这里 x 扮演时间轴，方程是 `u_x + u·u_y − λ·u_yy = 0`。）
2. **蒸馏生效**：学生 `loss_teach` 单调下降，最终与教师同量级。
3. **结构真的被"发现"了**：不能只用每层簇数作为硬判据。论文 Burgers 算例的第二层包含三类共享权重，并且还要求跨层的符号翻转、参数复用与排列关系。应同时检查 `structure.json` 的符号矩阵、替换簇中心后的误差，以及将关系矩阵固定后重训的结果。低步数试跑的簇数为 `[6,2,8,4,7,3,3,1]`，仅说明提取脚本可运行，不能宣称已复现论文结构。

---

## 5. 与上游源码的差异（早期快照）

此节保留初次试跑时的两处改动说明；**本次全量实验实际使用的完整差异**请看 [`../patches/upstream-local.patch`](../patches/upstream-local.patch)，不要仅根据下面两处推断全量运行条件。

### 5.1 修了一个必崩点：`Module/Training.py`

上游只给 `Flow_EXP.csv` 配了 `milestone` 字段，其余配置 `self.milestone = None`，
而 `train_adam()` 无条件构造 `MultiStepLR(milestones=self.milestone)`。torch 的 `MultiStepLR.__init__`
第一件事就是拿 `milestones` 去比较排序，`None` 迭代不了 → 直接抛异常，
于是**除 Flow 外的所有算例开箱即崩**。

补丁按作者原意处理：没有 `milestone` 就不建调度器。单就这项早期修复而言，没有 milestone 的 `MultiStepLR` 本来就是个空操作。当时改动 9 增 2 删；后续还有其他补丁，不能把这段当作当前完整 diff：

```bash
git -C /mnt/d/work/research/PINN/psi-NN/Psi-NN diff      # 查看
```

### 5.2 新增了一个低步数配置：`Config/Burgers_inv_distill_REPRO.csv`

从 `Burgers_inv_distill_EXP.csv` 逐行复制，**只改两个值**：

| 键 | EXP（原版） | REPRO（本复现） | 原因 |
|---|---|---|---|
| `train_steps` | 100000 | 10000 | 缩短到十几分钟 |
| `pace_record_gap` | 100 | 1000 | 少存 90% 的中间快照，loss 曲线不受影响（loss 每步都记录） |

原 `*_EXP.csv` 一个字都没改，`Results/` 目录也是新建的，和原版实验互不干扰。

---

## 6. 常见报错

| 现象 | 原因 / 处理 |
|---|---|
| `No module named pip` | 系统缺 `python3-venv`/`python3-pip`，venv 建出来没有 pip。`01_setup_env.sh` 会自动 apt 补齐；手动跑则是 `sudo apt-get install -y python3.12-venv python3-pip` |
| `SSL_ERROR_SYSCALL` / 连不上 pypi.org | 本机 WSL 访问不了 pypi.org，用镜像：`pip install -i https://mirrors.aliyun.com/pypi/simple <包名>` |
| `no kernel image is available` | torch 装成了 CPU 版或 CUDA 版本 < 12.8。重跑 `01_setup_env.sh`，确认输出里 `torch cuda` 是 `12.8` |
| `cuda avail: False` | 不是装错了 torch 就是驱动/WSL 的问题：Windows 侧执行 `wsl --update` 后重开 WSL |
| `ModuleNotFoundError: No module named 'Module'` | 工作目录不对，必须在 `Psi-NN` 根目录（有 `Module/`、`Config/` 那一层）下运行 |
| 结构提取报找不到 `.pth` | 训练没跑完，或 `--ckpt` 路径写错 |
| 觉得慢 | `/mnt/d` 是 9p（`msize=65536`），小文件 I/O 比 ext4 慢。可以 `cp -r .../Psi-NN ~/Psi-NN` 后 `REPO_DIR=~/Psi-NN bash 02_run_repro.sh`，代价是结果进了 WSL 里，Windows 侧看要绕一层 |
| 出图刷一堆 `findfont` 警告 | 本机没有 Times New Roman 字体（实测 0 个），回退到 DejaVu Sans。图正常，只是字体不是论文那套 |

---

## 7. 脚本清单

| 文件 | 作用 |
|---|---|
| `01_setup_env.sh` | 建 venv、装依赖、GPU + 二阶自动微分自检 |
| `02_run_repro.sh` | 一键：训练 → 结构提取，日志同时落盘到 `Results/_repro_logs/` |
| `run_train.py` | 训练入口，等价于 `Panel.py` 里的一行，但只跑一个算例 |
| `extract_structure.py` | `pic_parameter.ipynb` 第 2、4 单元的无头版（默认 Ward；`--linkage average` 对齐补充材料）+ 字母矩阵 + JSON |
| `evaluate_burgers.py` | 用 Cole–Hopf 解独立计算整域 MSE、相对 L2，并核对作者观测数据 |
| `evaluate_laplace.py` | 用解析解独立计算 Laplace 整域 MSE、相对 L2、边界与对称误差 |
| `run_reconstructed.py` | 把提取的符号矩阵作为固定参数共享关系，以可学习簇中心重训；不含论文后续人工块合并 |
