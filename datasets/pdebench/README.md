# PDEBench / DaRUS-2986 数据集说明

核查日期：2026-10-23。用户提供的 [DaRUS 链接](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) 对应 **PDEBench Datasets**，DOI `10.18419/darus-2986`。它是多个 PDE 数值模拟结果组成的基准集合，不是现场测量，也不是一份单独流场文件。

本次读取官方 Dataverse API：版本 **8.0**，发布日期 `2024-02-13`，336 个文件，合计 **3,902,903,970,421 字节**（约 3.90 TB / 3.55 TiB），数据许可 **CC BY 4.0**。这些是本次元数据快照，未来应再次读取核对；不能以代码 README 的旧体量覆盖实际目录。摘要见 [metadata-summary.json](metadata-summary.json)，逐文件名称、ID、大小、校验和和下载 URL 见 [catalog.csv](catalog.csv)。

## 包含哪些问题

下面按官方目录前两级汇总；GB/TB 为十进制，不是 GiB/TiB：

| 目录类别 | 文件数 | 合计 GB | 可用来研究什么 |
|---|---:|---:|---|
| 1D Advection | 8 | 65.864 | 波形输运、时变状态重建 |
| 1D Burgers | 12 | 98.796 | 非线性输运与扩散，可衔接已学问题 |
| 1D CFD | 6 | 94.377 | 一维可压缩流、激波等 |
| 1D diffusion-sorption | 1 | 4.217 | 扩散与吸附 |
| 1D ReactionDiffusion | 16 | 66.191 | 反应与扩散的竞争 |
| 2D CFD | 8 | 572.523 | 二维可压缩流 |
| 2D DarcyFlow | 5 | 6.554 | 多孔介质流；静态任务不直接提供时间历史 |
| 2D diffusion-reaction | 1 | 13.244 | 二维反应扩散 |
| 2D shallow-water | 1 | 6.626 | 浅水动力学 |
| 2D NS_incom | 275 | 2732.292 | 非定常不可压缩流，数据量最大 |
| 3D/Train | 3 | 242.221 | 三维可压缩流数据 |

官方描述是数据生成方式和研究用途，不保证所有参数、初边条件与我们的现有论文例子一致。不同方程的数据数量、空间维数、变量数和存储层次都可能不同。

## 数据怎么读，维度是什么意思

文件主要是 HDF5：可把它理解为一个含多个具名数组的容器，而不是 CSV 表。坐标数组、时间数组与物理场可能分开存。官方 [`PDEBench` 代码](https://github.com/pdebench/PDEBench)提供生成器、下载器和模型读取器，本机单独检出在 `upstream/`，精确提交记录在 [资料清单](../../shred/materials.json)。

根据官方 `pdebench/models/fno/utils.py` 的读取分支：

- 常见标量方程读取 `tensor`，原数组通常为 `[样本/轨迹, 时间, x(,y,z)]`。
- 可压缩流的变量分开读，例如 `density`、`pressure`、`Vx`，二维/三维还有其他速度分量；坐标可能叫 `x-coordinate` 等。
- 部分 `.h5` 使用分组存储，各组是一个样本；不能只依据后缀认定维度。
- 官方 FNO 读取器会把数组转为 `[样本, 空间..., 时间, 通道]`。SHRED 适配则通常需要 `[轨迹, 时间, 展平空间×变量]`，因此要明确 permute/reshape，不能直接复制一个网络的 DataLoader 给另一个网络。

上述是**源码格式说明**，不是宣称所有 336 文件已经逐个打开。本轮仅下载一个最小文件并校验完整性；本机现有环境缺少 h5py，尚未读出它的实际数组 shape。后续在已有 h5py 的实验机上用下述脚本列出字段和维度，不会把整份数组读入内存：

```bash
python datasets/pdebench/inspect_sample.py
```

没有 h5py 时该脚本只检查文件大小、MD5 和 HDF5 文件头，并明确报告“未检查 shape”。不会自动安装依赖。

## 本次下载了什么

- 官方原始元数据（本机 `dataset-metadata.json`，不提交其完整内容）；发布轻量摘要和 CSV 目录。
- 官方代码仓库（本机 `upstream/`，不嵌套提交）。
- `data/Sod6.hdf5`：文件 ID `133150`，4,948,776 字节，目录 `1D/CFD/Train/ShockTube`，MD5 `adb2d95bf0d48e03bc0d8f4a2cbcd1c6`。大小、校验和、HDF5 文件头均匹配。

Sod6 仅作为小型格式样本，不与 Ψ-NN 的 Burgers CSV 混用，也未作为完整 SHRED 训练集。全体 3.90 TB 未下载。已下载样本与原始数据被 Git 忽略；目录、校验和及小型检查报告可同步。

## 下一步选哪个子集

建议先核实一维 Burgers 或 Advection 的单一参数文件，选少量完整轨迹、适当降采样，再抽 3 个传感器构建时窗。这是后续候选，尚未启动大文件下载或训练。

本次目录中的 `1D_Burgers_Sols_Nu0.01.hdf5` 为 ID `281363`、8,232,968,312 字节（约 8.23 GB / 7.67 GiB）。官方旧下载示例曾使用其他文件 ID，因此实际下载前要按本次目录与 API 再核对，不能依赖旧教程编号。生成器的 multi 配置有 `nx=1024`、`dt_save=0.01`、`t∈[0,2]`、`numbers=10000`，仍应实际检查文件的数组与时间轴，不仅凭配置推断存储形状。

生成代码还使用 `epsilon/pi` 构造扩散通量；文件名的 Nu、文中黏度与其他项目的 λ 命名不能直接画等号，需同时核对方程定义、空间域、边界及保存版本。

SHRED 数据划分先做完整轨迹 train/validation/test，再构窗；归一化和 POD 基只用训练轨迹拟合，防止测试信息进入训练。HDF5 选择部分轨迹、时间和空间位置可以减少**内存**；本地切片不意味着远端只需下载这几条轨迹，网络下载体量要另核实是否支持可靠的远程切片。

来源：[官方数据页](https://doi.org/10.18419/darus-2986)、[元数据 API](https://darus.uni-stuttgart.de/api/datasets/:persistentId/?persistentId=doi:10.18419/darus-2986)、[官方代码与读取器](https://github.com/pdebench/PDEBench)、[PDEBench 论文](https://arxiv.org/abs/2210.07182)。公开引用时按数据页推荐格式引用数据作者、版本和 DOI；不要只引用自己的下载链接。
