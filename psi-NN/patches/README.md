# 运行补丁与中文教学注释

上游固定提交：`2c11ce6df1bcfcf630d05f162d5795d7fa4fdb43`，来源为 [ZitiLiu/Psi-NN](https://github.com/ZitiLiu/Psi-NN)，Apache-2.0。

`upstream-local.patch` 是已有实验机运行补丁；它修补调度器、缓存等运行行为。`learning-comments.patch` 是此次同步的中文教学注释，**以已应用前一个补丁的源码为基准**，不改变其可执行逻辑。它解释设备、张量形状、坐标求导、参数求导、Adam、蒸馏和共享结构。

如已有源码和依赖，在上游目录 `psi-NN/Psi-NN/` 中，对干净的上述固定提交按顺序操作：

```bash
git apply --check ../patches/upstream-local.patch
git apply ../patches/upstream-local.patch
git apply --check ../patches/learning-comments.patch
git apply ../patches/learning-comments.patch
```

已经应用运行补丁的实验机只需检查并应用教学补丁；不要重复应用运行补丁。若存在自己的未提交修改，先查看 `git status` 与 diff，保留现有工作；不要通过 reset 或强行覆盖来消除冲突。教学补丁是可选的，读源码时用，不要求为同一实验重新训练。

此次已在临时干净检出上验证两补丁依次可应用，并比较 Python AST：添加教学注释前后，13 个 Python 文件的可执行语法一致。两处重叠位置保留了实验机原有运行修补，再补充解释。

阅读顺序：`Module/PINN.py` → `Training.py` 中的 `mesh_init / net_f / net_b / train_adam`（先只看 Laplace）→ `PsiNN_laplace.py` → Burgers 逆问题与蒸馏。其余结构网络和 PINN-post 也有分支说明。

教学注释中不再使用 Mac 绝对路径。新增的 `repro/run_mac.py` 默认只做 100 步短程验证；使用现有环境手动运行，不自动安装环境或启动长程训练。不要把原版 `Panel.py` 当作短程入口，它会串行运行多组长程实验。
