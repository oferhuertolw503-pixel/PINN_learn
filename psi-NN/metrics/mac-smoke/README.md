# Mac 历史短程检查摘要

运行时间：2026-10-19；上传同步：2026-10-21。设备为 M5 MacBook Air、16 GB 统一内存、10 核 CPU、8 核 GPU；环境版本见 `environment.json`。

这些 JSON 来自短程功能与成本检查，**不是论文精度复现结果**，不合并进实验机的 `../phase1.csv`。原版 EXP 网络设置及损失函数与正式实验的 PAPER 配置、运行补丁不同。部分 CPU 短测曾同时运行，时间含预热，不能当作严谨的硬件性能对比。

| 文件 | 内容 |
|---|---|
| `Laplace_cpu.json` | 三模型各 100 步，10201 配点；独立评估为 101×101 网格 |
| `Laplace_mps.json` | 三模型各 20 步，使用 Metal GPU；包含后端启动与预热影响 |
| `Burgers_inv_distill_cpu.json` | 教师 20 步、学生 40 步 |
| `Poisson_cpu.json` | 三模型各 20 步，只判断优化过程能执行，不比较解析真值误差 |
| `Flow_cpu.json` | 三模型各 5 步，只判断 NS 损失和优化可执行，不代表完成 Flow 实验 |
| `structure_checks.json` | 未训练结构网络的对称性检查；非零误差否定无条件保证，不代表训练后不能逼近 |
| `environment.json` | 原运行包版本，不是需要另一台机器照装的依赖锁定文件 |

`mps_available=false` 出现在 CPU 摘要中是当时沙箱访问限制；能访问系统 GPU 的进程已实际通过 MPS 检查。`peak_process_rss_mib` 为进程历史峰值 CPU 驻留内存，不是 GPU 内存，也不是各模型独立的峰值。

新增的 `repro/run_mac.py` 由原 Mac 脚本调整为本仓库相对目录，默认 100 步；其新输出放在上游 `Results/mac_smoke/`，被忽略。新增的 `repro/check_structure.py` 输出到上游 `Results/structure_checks.json`。这两个入口可用 `REPO_ROOT` 指定已有上游目录，使用现有依赖，不负责安装。原始权重、完整日志与运行环境不在本目录中。
