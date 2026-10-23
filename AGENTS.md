# 两机协作规则

- 每轮开始先读取 GitHub 最新信息：检查本地状态，fetch 远端；干净且可快进时 `git pull --ff-only`，或通过 GitHub 工具读取 `main`。先读 README、`docs/learning-progress.md` 和本轮相关阶段文档。无法连接时说明未同步和当前基准，不把旧对话作为最新进度。
- Mac 用于论文/源码阅读、注释、小型检查和文档；Windows/WSL 实验机用于正式训练、评估和保存完整日志/权重。短程检查不冒充论文精度复现。
- 提交代码、配置、方法记录、小型指标摘要；不提交论文 PDF、上游独立检出、训练环境、原始大数据、checkpoint 和完整 Results。
- 作者预写结构、聚类中心替换、固定簇共享重训、自动跨层关系重建分别记录，不混称“完整自动发现”。
- 新实验使用 `docs/experiment-log-template.md` 记录代码版本、配置、数据划分、种子、评估口径与限制。失败结果保留，不用训练 loss 替代独立误差。
- SHRED/PDEBench 后续工作见 `docs/phase-2-shred.md` 与 `datasets/pdebench/README.md`。计划、短程检查、完整实验明确区分。
- 修改前同步；保留用户未提交修改；不强推，不因同步而重置用户工作。正式运行前确认代码版本。首次安装新工具或依赖遵循 README 的单独确认约定。
