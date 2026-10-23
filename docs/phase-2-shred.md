# SHRED：从稀疏传感器到完整场，下一条实战链

更新：2026-10-23。是后续学习方向和初步流程检查，不代表第一阶段 Ψ-NN 已完成，也不代替其正式实验记录。开工时已读取 GitHub 最新 `d1639e7`：Poisson 已完成无衰减/正文启发调度的各 30k 比较，Flow 已完成三模型各 1k pilot，详见 [第一阶段记录](phase-1-psi-nn.md)。

## 先理解它训练什么

PINN 常把坐标送入网络，借助 PDE 导数残差与边界/观测约束训练。基本 SHRED 则给网络一段传感器历史，让它预测该段历史最后时刻的完整状态；训练标签是离线获取的完整场，通常不包含 PINN 式的 PDE 残差。两者都做反向传播，但输入、监督和用途不同。

```text
离线准备：完整场序列 u[轨迹,时间,空间(与变量)]
  ↓ 先划分完整轨迹，保留验证/测试
仅用训练集拟合归一化（以及可选 POD 基）
  ↓ 选出少量固定传感器位置，抽取这些位置的时间读数
切窗口：[batch,历史长度L,传感器数S]
  ↓ LSTM 逐时处理，把一段观测汇成隐藏状态 [batch,H]
浅层 MLP 解码：[batch,H] → [batch,完整场维数P]
  ↓ 与窗口最后时刻的完整场标签计算误差
zero_grad → backward → Adam.step
  ↓ 验证集选择模型，独立轨迹上评分
在线使用：给定最近L步真实传感器读数 → 重建当前完整场
```

例如一个 batch 有 64 个窗口、每个窗口 8 步、每步 3 个读数，输入就是 `[64,8,3]`。原始模型的 `batch_first=True` 表示第一维为 batch。`h_out[-1]` 取最后一层 LSTM 的最终隐藏状态，经过几个 Linear/ReLU 后得到 `[64,P]`。

LSTM 可以先当成“逐时更新记忆的神经网络”。它的隐藏状态由数据和权重计算；初始 `h_0/c_0` 通常为零，不是完整物理场的真实初值。理解序列维、隐藏状态和最后状态如何送入 MLP 后，再补门控公式即可。

基本 SHRED 的“重建”不等于“预测未来”。输入窗口已含 t 时刻观测、输出 u(t)，是在补全当前状态；若要输出 t+1 或在没有后续观测时滚动预测，需要另定义预测标签/传感器预测器，并单独评估误差增长。不能用当前时刻的新传感器读数评价“完全无观测的长期预测”。

训练阶段必须有全场标签或替代监督。只给三条传感器曲线，并不能让未经训练的 SHRED 自动知道任意新流场。少量传感器能否恢复某种状态，要由具体动力学、采样间隔、传感器位置、噪声和独立实验来验证。

## 对照下载的代码，按执行顺序读

原始代码固定在 [Jan-Williams/pyshred@e5b1222](https://github.com/Jan-Williams/pyshred/tree/e5b122259f59cb0d4a91fa60da812349616380b3)。完整提交与资料校验见 [清单](../shred/materials.json)。

| 步骤 | 原始文件与对象 | 重点看什么 |
|---|---|---|
| 读场数据 | `processdata.py: load_data` | SST 的矩阵转置、移除陆地；时间与空间轴不能反 |
| 选择传感器 | `example.ipynb`、`reconstructions.py` | 固定位置从全场中抽列，QR 是另一种位置选择方法 |
| 划分与归一化 | notebook 的 `train_indices`、`MinMaxScaler` | 缩放器拟合了什么数据；没有用测试标签拟合 |
| 切时间窗 | `all_data_in[i] = transformed_X[i:i+lags, sensor_locations]` | 每个窗口包含哪些时刻、是否跨轨迹边界 |
| 对齐标签 | `train_indices + lags - 1` | 预测窗口末时刻，而非默认预测下一步 |
| 数据打包 | `processdata.py: TimeSeriesDataset` | 返回一对 `(sensor_window, full_field)` |
| 前向 | `models.py: SHRED.__init__/forward` | LSTM → 最后一层最终隐藏状态 → Linear/ReLU 解码器 |
| 训练 | `models.py: fit` | DataLoader、MSE、Adam、验证相对误差、最佳权重恢复 |
| 还原物理量与评分 | notebook 的 `inverse_transform` | 用原单位计算误差，明确相对 L2 的分母 |
| 未来预测扩展 | `forecasts.py`、`models.py: forecast` | 将传感器预测器与场重建器分开，而非把重建当未来预测 |

原始 `example.ipynb` 对一条 SST 记录中的窗口随机划分，更接近同一记录内的重建评估。相邻窗口可能重叠，因此这种指标不能直接证明未来时间外推或新参数轨迹泛化。自己的跨轨迹实验先按完整轨迹划分，再切窗；只在训练集拟合缩放、选取 POD 模式和调传感器。

作者 `fit` 在第 1 轮及之后每 20 轮检查验证误差，patience 计的是验证检查次数，而非每一轮。新小脚本为了便于观察，每轮做验证，并明确记录与作者训练器的差异；模型类仍直接来自原始源码。

## SHRED-ROM 多出来的一步

当 P 很大时，最后一层直接输出完整场会很大。SHRED-ROM 先从**训练场**求一个秩为 r 的 POD/SVD 基，把完整场压缩成 r 个系数，让网络预测系数，再乘固定基重建完整场。

```text
传感器历史 → LSTM → MLP → r个POD系数 → 固定训练基 → P维完整场
```

POD 基不是从测试场一起拟合的。原始 SHRED 可以不带 POD；POD 不是 LSTM 的另一种叫法。下载的 SHRED-ROM 入口是 `utils/models.py` 的 `SHRED/fit`、`utils/processdata.py` 的 `Padding/TimeSeriesDataset`，以及 `DoubleGyreFlow.ipynb`。后者是后续流体练习候选，不在今天宣称已训练。

当前 PySHRED 软件包封装了预处理、序列模型和解码器，读懂短版后再看 `examples/basic_shred_sst.ipynb`。其 `setup.py` 含 `numpy<2.0` 和额外依赖；现有 Mac 试跑环境的 NumPy 为 2.3.5，因此本次只下载阅读，未直接安装，避免改变现有 PINN 环境。原版/SHRED-ROM 的隐状态目前只分 CPU/CUDA，也不能仅写 `.to('mps')` 就假设 Apple GPU 可直接跑。

## 已完成与接下来做什么

| 项目 | 状态 |
|---|---|
| 三篇论文、三套源码 | 本地已下载，固定源码提交并记录文件哈希 |
| 原始 SST 实际文件 | 已下载，LFS SHA-256 匹配，MAT 元数据确认；尚未训练 |
| 原作者模型的小型链路 | 已在 32 条解析热方程轨迹上训练 10 轮并测试；只是 smoke |
| PDEBench | 元数据、336文件目录、官方读取代码和一个最小文件已取到；完整场训练尚未做 |
| 完整 SHRED 论文实验、SHRED-ROM、PDEBench-SHRED | 计划阶段，未完成 |

本轮小型脚本见 [smoke_original.py](../shred/repro/smoke_original.py)，结果见 [摘要](../shred/metrics/synthetic_heat_smoke.json)。测试相对 L2 约 0.5645，误差仍明显；不能据此宣传高精度或工程有效。

后续按实际操作走：先逐行解释这个小脚本的输入窗口、目标和反传；再读 SST notebook 并明确评估划分；之后让实验机选择 PDEBench 一个小数据子任务，做数据适配、少量轨迹/空间降采样试跑。正确性检查通过后，再增加 POD、噪声和跨参数检验。第一份真实数据试验优先沿用已熟悉的一维问题；Sod6 只作格式样本，其单独文件是否具有足够独立轨迹还需读取，不预设适合完整 SHRED 训练。

评价时同时报告训练均值常量基线、仅当前读数的 MLP，以及有历史的 SHRED；条件允许时再加 POD/插值基线。固定轨迹划分、传感器、归一化、预算和模型选择规则；全场误差和未观测点误差分别算，避免已知传感器点让指标看起来更好。

## 与 PDEBench 怎么接

PDEBench 是[公开数据来源](../datasets/pdebench/README.md)，不是 SHRED 自带数据，也不是 Ψ-NN 现有的少量 CSV。时变样本适配成 `[轨迹,时间,空间×变量]` 后可抽传感器构窗；静态 Darcy 样本没有真实时间历史，不应把空间轴改名“时间轴”来套 LSTM。

还要分清任务：PDEBench 的 FNO/U-Net 基线通常做时间推进或参数到解映射，SHRED 若每步有新的传感器观测，就是状态重建。两者输入信息不同，不能把误差直接并排后宣称方法优劣。Burgers 参数的文件名、生成代码中系数与边界条件也应逐项核实，不能把它自动等同于 Ψ-NN 的实验设定。

来源：[原始论文](https://arxiv.org/abs/2301.12011)、[SHRED-ROM](https://arxiv.org/abs/2502.10930)、[PySHRED](https://arxiv.org/abs/2507.20954)、[PDEBench 数据 DOI](https://doi.org/10.18419/darus-2986)。本地 PDF 仅供阅读，不纳入 Git 提交。
