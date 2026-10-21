# 第一阶段指标来源

[`phase1.csv`](phase1.csv) 是方便 Mac 阅读与绘图的脱敏摘要；它不是原始训练日志。每行都使用实验机最终 checkpoint 在 200×200 网格上独立评估。空的 `trainable_parameters` 表示聚类中心直接替换并未完成结构重训；`nu_estimate_rounded` 只保留工作记录中的黏度近似值。

在已有上游源码、补丁、配置、checkpoint 和 Python 依赖的实验机上，从 `psi-NN/Psi-NN/` 目录运行以下命令可核对主要行（不安装依赖）：

```bash
python ../repro/evaluate_laplace.py --question Laplace_distill --ini PAPER --grid 200
python ../repro/evaluate_laplace.py --question Laplace --ini PAPER_COMPARISON --grid 200
python ../repro/evaluate_burgers.py --question Burgers_inv --ini PAPER_BASELINE --grid 200
python ../repro/evaluate_burgers.py --question Burgers_inv --ini PAPER_AUTHOR --grid 200
```

`average_cluster` 与 `ward_cluster` 行还需向 `evaluate_laplace.py` 传入对应 `Results/Laplace_distill_PAPER/Cluster_*/Laplace_distill_PAPER_structure.json` 的 `--structure`。原始 JSON 会含实验机绝对 checkpoint 路径，所以不直接复制进公开仓库。当前 CSV 仅一颗随机种子，不能用于显著性判断。
