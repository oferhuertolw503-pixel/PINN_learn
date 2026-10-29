# 两机学习进度交接

同步日期：2026-10-29。当前用户基础为 Karpathy Zero to Hero 前两集，学习目标是从源码明确理解每个 PDE 怎么准备输入、构造网络与损失、求导和更新参数；希望以实战贯通，暂不需要完整课程体系。

## 2026-10-29 Mac 目录与环境迁移

- 主开发副本已迁至新的 Project 目录，源码、论文、SST/PDEBench 本地材料已复制校验；原目录仅作备份，用户明确不保留其中的 PPT。
- 用户已删除 Conda。核查旧 Mac 虚拟环境原本使用 Codex 运行时 Python，并不依赖 Conda；本次进一步用已安装的 uv 获取独立 Python 3.12.14，在新仓库创建不继承系统包的 `.venv`。
- 八个直接依赖保持原版本，32个实际安装包的完整版本见 [requirements-mac-learning.txt](../requirements-mac-learning.txt)。不再依赖旧目录、Conda 或 Codex Python；WSL CUDA 环境未改动。
- 已验证全部关键包从新 `.venv` 导入、依赖一致性检查通过、Metal 后端可用；新环境下 SHRED 两轮合成数据检查和 Laplace 两步 PINN 均执行成功，仅作为迁移验证。
- 日常激活方式和复建命令见 [Mac 环境说明](mac-environment.md)。环境目录和本机检查日志不提交；正式实验摘要不改写。当前未安装 h5py 或 PySHRED 软件包。

## 2026-10-23 新接续

- 本轮先读取 GitHub 最新内容，同步到实验机提交 `d1639e7`；Poisson 已有两种学习率方案各30k比较，Flow 已有三模型各1k pilot。正式数值继续以 [第一阶段记录](phase-1-psi-nn.md) 为准，下面早期 Mac 检查不覆盖这些新结果。
- 将“本项目每轮先读取 GitHub README、进度和相关阶段文档”的约定写入 [AGENTS.md](../AGENTS.md)，断网时说明未同步，保留本地未提交工作。
- 新增后续方向 [SHRED](phase-2-shred.md)：已下载原始 SHRED、SHRED-ROM、PySHRED 三套代码和三篇论文，补取原始 SST 实际数据并验证 LFS SHA-256；来源与精确提交见 [材料清单](../shred/materials.json)。资料本体留本机，不入 Git。
- 已直接调用原作者 SHRED 模型，在32条解析热方程轨迹上跑通3个传感器、8步窗口到64点全场的训练/验证/测试链。10轮CPU smoke 的测试相对L2约0.5645，仍有明显误差；不是 SST/PDEBench 或论文精度复现。已有脚本和小摘要可同步。
- 用户给出的 DaRUS 数据是 [PDEBench](../datasets/pdebench/README.md)：版本8.0、336文件、约3.90TB、CC BY 4.0。已保存官方目录、下载官方代码与4.95MB最小样本；全量未下载。样本MD5和HDF5签名匹配，本机缺h5py，未宣称读取其数组形状，也未安装新依赖。
- 接下来先逐行理解最小 SHRED 链路，再检查 SST 的时间窗及划分，然后在实验机选择 PDEBench 的一个一维子任务接入。跨轨迹与未来时间预测分别评价，归一化/POD只用训练集。

## Mac 已完成

- 阅读作者固定提交，写成 [论文与源码逐步对照](论文与源码逐步对照.md)，包括教师、学生、聚类、关系矩阵与结构重训；明确作者预写网络与自动发现过程的区别。
- 为入口、普通 MLP、训练器、四类结构网络和四种 PINN-post 添加中文注释，作为 [可叠加的注释补丁](../psi-NN/patches/README.md) 同步。已验证仅增加解释，不改变实验机运行补丁后的可执行逻辑。
- 整理 [从 micrograd 读懂 PINN](从micrograd读懂PINN.md)，重点说明张量形状、设备选择、两种求导和训练循环。
- 在 Mac 上做 CPU 短程检查：Laplace 三模型各 100 步、Burgers 教师 20/学生 40 步、Poisson 三模型各 20 步、Flow 三模型各 5 步；Laplace MPS 三模型各 20 步。这些只证明执行分支可运行，不验证论文精度；这是早期历史记录，实验机后来已完成 Flow 1k pilot，见最新第一阶段记录。
- 验证随机初始化时结构对称性：Laplace 的 y 偶对称、Poisson 的坐标交换对称成立；Burgers 输出 bias 和 Flow 最后自由混合层使相应输出对称性并非对任意参数严格成立。见 [小型摘要](../psi-NN/metrics/mac-smoke/README.md)。

Mac 短程运行使用原版可执行逻辑和 EXP 配置，早于此仓库的实验机同步；其 Python/PyTorch、网络宽度、步数和评估网格与正式复现不同。原日志、论文、截图和权重保留本地，不提交。原有正式记录及 `phase1.csv` 不因此次同步改写。

## 下一次接着哪里学

1. 建议补 Karpathy **Building makemore Part 2: MLP**（全系列第 3 集），重点为 MLP、张量形状、学习率和评估；第 4 集的激活/梯度诊断按实际问题选看。
2. 阅读带注释的普通 MLP 和 Laplace 分支，能解释 `[N,2] → [N,1]`、配点/边界点、方程残差和四步更新循环。
3. 能区分 `autograd.grad(u, x)` 对坐标求导与 `loss.backward()` 对权重求梯度，再把同一流程与实验机的 CUDA 版本对照。
4. 再学 Burgers 的未知系数优化、教师/学生蒸馏、聚类和共享结构；随后扩展到 Poisson、Flow。

这是阅读接续位置，不是已完成的学习认定，也不是新训练任务。研究实验的进度和下一项验证继续以 [第一阶段正式记录](phase-1-psi-nn.md) 为准；本次不启动实验机长程训练，不重复安装环境。

## 本次交接边界

同步的是文档、注释补丁、两个轻量辅助脚本和脱离本机绝对路径的历史摘要。实验机的原补丁、配置、正式指标保持不变。下一轮修改前先同步最新提交；有新实验时用 [实验记录模板](experiment-log-template.md) 关联代码版本与指标。
