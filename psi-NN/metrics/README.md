# 第一阶段指标来源

[`phase1.csv`](phase1.csv) 是方便 Mac 阅读与绘图的脱敏摘要；它不是原始训练日志。每行都使用实验机最终 checkpoint 在 200×200 网格上独立评估。空的 `trainable_parameters` 表示聚类中心直接替换并未完成结构重训；`nu_estimate_rounded` 只保留工作记录中的黏度近似值。

在已有上游源码、补丁、配置、checkpoint 和 Python 依赖的实验机上，从 `psi-NN/Psi-NN/` 目录运行以下命令可核对主要行（不安装依赖）：

```bash
python ../repro/evaluate_laplace.py --question Laplace_distill --ini PAPER --grid 200
python ../repro/evaluate_laplace.py --question Laplace --ini PAPER_COMPARISON --grid 200
python ../repro/evaluate_burgers.py --question Burgers_inv --ini PAPER_BASELINE --grid 200
python ../repro/evaluate_burgers.py --question Burgers_inv --ini PAPER_AUTHOR --grid 200
python ../repro/evaluate_poisson.py --ini REPRO --grid 200
python ../repro/evaluate_poisson.py --ini PAPER_SCHEDULE --grid 200
python ../repro/evaluate_flow.py --ini PILOT
```

`average_cluster` 与 `ward_cluster` 行还需向 `evaluate_laplace.py` 传入对应 `Results/Laplace_distill_PAPER/Cluster_*/Laplace_distill_PAPER_structure.json` 的 `--structure`。原始 JSON 会含实验机绝对 checkpoint 路径，所以不直接复制进公开仓库。当前 CSV 仅一颗随机种子，不能用于显著性判断。

Poisson 行按训练代码实际求解的 `Δu=f`，使用与该源项、零边界一致的解析解计算；作者 `net_global()` 中的另一参考场与此 PDE 不一致，因此不拿它当真值。`Poisson_REPRO` 保留作者 `Poisson_EXP.csv` 的无学习率衰减设置，仅放宽中间权重保存间隔。`Poisson_PAPER_SCHEDULE` 是把正文所述的 5k/15k 步衰减解释为每次乘 0.2 的敏感性对照，最终学习率为 `4×10⁻⁵`；论文措辞和仓库配置存在差异，不能把它冒称唯一的官方设置。两组的边界、交换对称、作者参考式误差及训练秒数见 [`poisson_diagnostics.csv`](poisson_diagnostics.csv)。

[`flow_pilot.csv`](flow_pilot.csv) 单独记录圆柱绕流的压力 `p`、水平速度 `u`、竖直速度 `v`。`expected_y_parity_mse` 比较预测与关于中心线反射后的预测，`p/u` 应为偶函数、`v` 应为奇函数；很小的对称误差本身不代表场精度好。压力的 `pressure_gauge_aligned_mse` 仅在 `p` 行给出，用于移除一个全场常数偏移。Flow 的 1,000 步只是约为作者配置 200,000 步的 0.5%：CSV 内部标签未直接进入方程点损失，但其坐标参与训练，且文件来源未核实，因此这些是同域试跑数值，不是独立工程验证。
