# SHRED 材料与实战入口

更新：2026-10-23。这里的 SHRED 指 **SHallow REcurrent Decoder**，即浅层循环解码器。已准备原始 SHRED、SHRED-ROM 和当前 PySHRED 三套资料；先读原始版本的短代码，再扩展到压缩重建和工具库。[后续工作与源码链路](../docs/phase-2-shred.md)。

## 本地已下载

| 本地目录/文件 | 来源与用途 |
|---|---|
| `upstream/original-shred/` | [Jan-Williams/pyshred](https://github.com/Jan-Williams/pyshred)，原始论文代码；优先读 `models.py`、`processdata.py` 和 `example.ipynb` |
| `upstream/shred-rom/` | [MatteoTomasetto/SHRED-ROM](https://github.com/MatteoTomasetto/SHRED-ROM)，POD 压缩与跨参数场重建 |
| `upstream/pyshred/` | [pyshred-dev/pyshred](https://github.com/pyshred-dev/pyshred)，当前工具包及示例，后续再看其封装 |
| `papers/sensing-with-shallow-recurrent-decoders.pdf` | [原始论文，arXiv:2301.12011](https://arxiv.org/abs/2301.12011) |
| `papers/shred-rom.pdf` | [SHRED-ROM，arXiv:2502.10930](https://arxiv.org/abs/2502.10930) |
| `papers/pyshred.pdf` | [PySHRED，arXiv:2507.20954](https://arxiv.org/abs/2507.20954) |
| `data/SST_data.mat` | 原始 SHRED 的 SST 示例，实际数据 187,391,805 字节，已核对其 Git LFS SHA-256 |

三个上游独立检出、论文和原始数据都被 Git 忽略，保留本机。共享仓库同步的是来源、固定提交、哈希、下载工具、自己的流程脚本和摘要，不重新发布上游源码或大数据。精确版本见 [materials.json](materials.json)。

原始仓库自带的 `Data/SST_data.mat` 是 **Git LFS 指针**，不能直接交给 `loadmat`。本机已将实际文件放在本目录 `data/` 下，保留作者仓库不改动。后续运行原例时，需让读取入口指向这个实际文件，或在独立工作副本中放入数据。MAT 文件的 `Z` 为 `[64800,1400]` double，转置后是 1400 个时间样本、64800 个空间位置，作者 `load_data('SST')` 还会移除陆地区域。187 MB 是磁盘文件大小；完整解压数组约 726 MB，多份副本会进一步占用内存。

## 已跑通的小链路

已有环境中，在共享仓库根目录运行：

```bash
python shred/repro/smoke_original.py --epochs 10 --device cpu
```

该入口直接导入固定提交的原始 `models.SHRED`，用自行生成的 32 条解析热方程轨迹；每条 64 个时间、64 个空间点，3 个固定传感器，窗口长度 8，按完整轨迹拆成 24/4/4 条训练/验证/测试。只有训练轨迹用于拟合归一化，验证集选 checkpoint，测试集最后评分。

这次 CPU 检查的测试物理场 MSE 为 `0.210079`、相对 L2 为 `0.564465`；训练均值常量基线 MSE 为 `0.621175`。误差仍大，只证明原模型的前向、反传和重建链路可执行。不是原论文、SST、PDEBench 或湍流实验的结果；没有验证传感器数量的充分性或跨分布泛化。记录见 [metrics/synthetic_heat_smoke.json](metrics/synthetic_heat_smoke.json)。

输出到 `shred/Results/`，由 Git 忽略。不自动启动长程训练，不安装新环境。已有 CUDA 环境的实验机可显式指定 `--device cuda`；原作者类不处理 MPS 隐状态迁移，因此当前小脚本只提供 CPU/CUDA。

## 另一台电脑怎样取相同材料

`fetch_materials.py` 仅使用 Python 标准库和 Git。默认不联网下载，只列出清单：

```bash
python shred/fetch_materials.py
python shred/fetch_materials.py --codes --papers
python shred/fetch_materials.py --sst
python shred/fetch_materials.py --pdebench-code --sample
```

`--sst` 约 187 MB；`--sample` 只取 4.95 MB 的 PDEBench Sod6 文件，不取整个 PDEBench。代码检出使用记录的提交；下载文件验证字节数与 SHA-256。arXiv 无版本号 PDF 链接将来可能更新，因此下载与记录哈希不同会停止，需人工核实新版本后更新清单，不能把新 PDF 当成当前阅读版本。

三套代码都声明 MIT 许可，保留各自来源与 LICENSE；这不等于对其示例数据作新的授权。PDEBench 数据另按其 CC BY 4.0 许可及 DOI 引用。
