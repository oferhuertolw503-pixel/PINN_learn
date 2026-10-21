# PINN 学习与工程化实践

这是我学习物理信息神经网络（PINN）的公开工作仓库。目标不是只跑通一篇论文，而是逐步弄清方法的适用范围，最终把经过验证的做法用于具体工程问题。

当前为第一阶段：精读并独立核验 [Ψ-NN 论文](https://doi.org/10.1038/s41467-025-64624-3) 中的知识蒸馏、权重聚类和网络重构。实验结果是**个人复现**，不代表论文作者的官方结果；“作者预写 Ψ-NN”与“从学生网络自动发现结构”在本仓库中严格区分。

## 两台电脑如何协作

| 设备 | 主要工作 | 通过 Git 同步 |
|---|---|---|
| Windows + WSL 实验机 | 训练、评估、保存完整日志与权重 | 代码、配置、方法记录、小型结果摘要 |
| Mac 阅读机 | 精读论文、阅读与修改代码、整理解释 | 同上；不要求具备 CUDA 或重复训练 |

`*.pth`、完整 `Results/`、本机环境和原论文 PDF 不提交。要在 Mac 上查看实验结论，先读 [第一阶段记录](docs/phase-1-psi-nn.md)；需核对原始权重时再回到实验机。本机仍保存未删改的详细原始工作记录，公开版本只收录与复现有关且适合发布的内容。

## 仓库导览

| 路径 | 内容 |
|---|---|
| [`docs/phase-1-psi-nn.md`](docs/phase-1-psi-nn.md) | 已做实验、真实指标、未完成部分、下一步判据；含英文术语 |
| [`docs/learning-progress.md`](docs/learning-progress.md) | 两机学习进度交接、Mac 已完成内容与接下来阅读的位置 |
| [`docs/从micrograd读懂PINN.md`](docs/从micrograd读懂PINN.md) | 面向 Zero to Hero 前两集基础的源码阅读说明 |
| [`docs/论文与源码逐步对照.md`](docs/论文与源码逐步对照.md) | 论文步骤与原版源码的函数对应、实现差异和 Mac 短程检查 |
| [`psi-NN/repro/`](psi-NN/repro/) | 自己编写的训练入口、结构提取、独立评估与受约束重训脚本 |
| [`psi-NN/configs/`](psi-NN/configs/) | 个人实验使用的配置快照；文件名区分试跑、论文表格与敏感性检查 |
| [`psi-NN/patches/`](psi-NN/patches/) | 相对于作者仓库固定提交的本机修补，便于第二台机器理解改动 |
| [`psi-NN/metrics/phase1.csv`](psi-NN/metrics/phase1.csv) | 脱离本机绝对路径的小型机器可读指标摘要 |
| `psi-NN/Psi-NN/` | 作者源码的独立检出目录，**不属于本仓库**，被 `.gitignore` 排除 |

作者源码来自 [ZitiLiu/Psi-NN](https://github.com/ZitiLiu/Psi-NN)，本次复现固定参考提交 `2c11ce6df1bcfcf630d05f162d5795d7fa4fdb43`。本仓库中我原创的代码、配置和文档按 [Apache-2.0](LICENSE) 发布；上游补丁的来源与改动见 [NOTICE](NOTICE)。作者源码保留其原有许可，论文 PDF 与训练产物不随仓库发布。

## 实验机的复建顺序

1. 获取本仓库，并在 `psi-NN/Psi-NN/` 单独检出上游固定提交；不要把上游目录当成本仓库普通文件提交。
2. 对干净的上游检出应用 [`psi-NN/patches/upstream-local.patch`](psi-NN/patches/upstream-local.patch)，再将 [`psi-NN/configs/`](psi-NN/configs/) 中选定的 CSV 放入上游 `Config/`。补丁仅适用于上述提交；先用 `git apply --check` 核查。
3. 在已有且满足依赖的 Python 环境中，从上游代码目录运行 [`psi-NN/repro/`](psi-NN/repro/) 的脚本。当前训练使用 WSL CUDA；**Mac 的日常阅读与改动不需要安装训练依赖**。
4. 将评估得到的数值和条件写入公开记录，并在实验机保留原始 checkpoint、日志及数据。不要只凭训练 loss 宣称复现成功。

首次搭建新机器的工具与依赖安装需要单独确认；本仓库不自动运行安装脚本。现有 [`01_setup_env.sh`](psi-NN/repro/01_setup_env.sh) 是早期本机环境记录，不是跨平台初始化入口。

协作时，Mac 负责提交笔记/代码修改，实验机在新实验前同步这些提交；实验机只提交经核对的配置、评估摘要和必要图表，不提交 checkpoint。每次新实验按 [实验记录模板](docs/experiment-log-template.md) 记录代码提交、配置、种子与评估口径，避免两台机器上的同名实验含义不同。

GitHub 以 **commit → push → pull** 为同步单位，不是两台电脑对同一文件的实时协同编辑；实验机运行前先确认已经拿到 Mac 的最新提交，避免代码版本与指标对不上。

Mac 的中文教学注释单独保存在 [`learning-comments.patch`](psi-NN/patches/learning-comments.patch)，在现有运行补丁之后可选应用；[补丁说明](psi-NN/patches/README.md)给出顺序与校验方法。注释不改变可执行逻辑。Mac 的历史短程摘要在 [`metrics/mac-smoke/`](psi-NN/metrics/mac-smoke/README.md)，与实验机正式指标分开保存。

## 工程化目标

第一阶段先建立可信基线与方法边界；后续要在明确 PDE、几何、边界条件、观测数据及误差/延迟目标后，比较传统数值法、普通 PINN、显式物理约束与完整自动结构 Ψ-NN。**目前尚不能宣称自动结构发现已复现，更不能直接宣称可工程部署。**

引用：Liu 等，*Automatic network structure discovery of physics informed neural networks via knowledge distillation*, Nature Communications 16, 9558 (2025), DOI: [10.1038/s41467-025-64624-3](https://doi.org/10.1038/s41467-025-64624-3)。
