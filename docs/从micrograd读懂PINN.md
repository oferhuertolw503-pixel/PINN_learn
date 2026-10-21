# 从 micrograd 读懂这份 PINN 代码

> 同步说明（2026-10-21）：本文记录 Mac 在 2026-10-19 的学习/短程检查，源码分析基于未应用实验机运行补丁的作者提交。它不替代 [第一阶段正式记录](phase-1-psi-nn.md)。代码导航按函数名查找；源码行号会随补丁改变。公开仓库不附作者源码，需按根目录 README 独立检出。中文注释补丁的用法见 [补丁说明](../psi-NN/patches/README.md)。

你已经知道“参数 → 预测 → 损失 → 反向传播 → 更新”。读这个仓库不需要重新学一遍神经网络，先把这条链迁移到 PyTorch，再增加“用导数检查方程”这一步即可。

CUDA 这一关很短：`device` 表示计算位置，`.to(device)` 把张量或模型放过去。`cuda` 对应 NVIDIA GPU，`mps` 对应 Apple GPU，`cpu` 就是 CPU。模型和参与同一次运算的数据要在同一设备。这里没有要求你编写 CUDA 内核，CPU 就能开始。

先打开 [PINN.py](https://github.com/ZitiLiu/Psi-NN/blob/2c11ce6df1bcfcf630d05f162d5795d7fa4fdb43/Module/PINN.py)，它相当于 micrograd 中的 MLP。`nn.Linear` 管权重与偏置，`nn.Tanh` 管非线性，`nn.Sequential` 依次执行这些层，`nn.Module` 自动收集参数。写 `net(xy)` 就是在算前向预测，不是在训练。

接着需要看懂形状。取 N 个二维位置时，坐标表是 `[N,2]`；第一列是 x，第二列是 y，输出表 `[N,1]` 存每个位置的预测 u。`x[:,0:1]` 保留列维度；`reshape(-1,1)` 把数排成一列；`cat(...,dim=1)` 拼列。看到一个操作，先能回答“输入、输出有几行几列”就够了。

然后看 [Training.py](https://github.com/ZitiLiu/Psi-NN/blob/2c11ce6df1bcfcf630d05f162d5795d7fa4fdb43/Module/Training.py) 的 `mesh_init → net_f → net_b → train_adam`。先只跟 Laplace 分支，暂时跳过 Flow、inv 和 distill。`self` 是同一次实验共享的状态；CSV 只是把宽度、步数等从代码挪到了表格，不是新的数学概念。

PINN 和你见过的有标签学习有一个关键区别：内部坐标可以没有真实 u。网络先预测 u，再检查它是不是接近满足 `u_xx+u_yy=0`。所以需要偏导数：固定 y，研究 u 随 x 的变化，叫 u_x；再对 x 求一次导，叫 u_xx。二维 Laplace 算子就是 u_xx+u_yy。边界条件则告诉网络区域边缘应该输出什么，帮助选择具体解。

这里有两种求导，务必分清：

```text
坐标 (x,y) → u_theta(x,y)
                  ↓ 对坐标求导：autograd.grad
              u_xx、u_yy
                  ↓ 代入方程并平方平均
              PDE loss + 边界 loss
                  ↓ 对权重求梯度：loss.backward()
              每个参数的 .grad
                  ↓ optimizer.step()
              更新权重，坐标保持原样
```

`requires_grad=True` 表示记录求导所需的运算，不等于“优化器会更新它”。更新哪些变量由传给优化器的参数列表决定。这里坐标需要梯度来计算物理导数，但没有交给优化器去更新。

`create_graph=True` 是为了让“求导这件事”也保留可求导的计算过程：否则后面求二阶导，以及将 PDE loss 对权重反传，会丢掉需要的路径。`retain_graph=True` 则是保留已经使用的图供再次求导。理解区别即可，不必把两个开关加到所有普通训练代码中。参考 [PyTorch autograd.grad 文档](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)。

损失方面只需掌握均方误差：`((预测-目标)**2).mean()`。Laplace 中，方程残差的目标是零，边界预测的目标是已知边界值。两项加起来得到一个标量；`loss.backward()` 算梯度，Adam 使用梯度和历史统计更新参数。Adam 不替你定义目标，也不保证自动收敛，学习率仍然需要检查。

看损失下降时，再用独立网格检查预测误差。训练配点上方程残差小，并不单独证明整个区域的解都准确；边界误差也要一起看。对这个有解析解的 Laplace 例子，可以直接与 `u=x³-3xy²` 比较。原来的短程脚本已在 summary 中记录全场 MSE 和相对 L2，二者不是同一个量。

这些理解后再看蒸馏：把学生的监督目标换成教师预测，再加一个参数规模的惩罚项。最后看结构网络：同一层被两个分支重复调用，就共享参数；两个分支相加或相减，可以让输出保持某种对称性。它们最终仍使用相同的反向传播和参数更新操作。

你若想继续看 Karpathy，最有帮助的是下一集 **Building makemore Part 2: MLP（全系列第 3 集）**，关注多层网络、训练循环、学习率和模型评估，字符 embedding 的任务细节可先放下。**Part 3: Activations & Gradients, BatchNorm（第 4 集）** 可在遇到梯度太小、训练不稳定时选看激活值、初始化与梯度诊断部分；当前仓库没有 BatchNorm，不必先学会把它加进来。视频名称和内容可核对 [作者课程页](https://karpathy.ai/zero-to-hero.html)。

为了当前任务，暂时不必追后面的 WaveNet、GPT 或 tokenizer。课程以外真正要补的是上面这些张量操作、PyTorch 的层与优化器、偏导/二阶导、边界条件、输入求导和参数求导的区别。

可以先手算 `u=x³+xy²`，得到 `u_x=3x²+y²`、`u_xx=6x`、`u_y=2xy`、`u_yy=2x`，再用 `autograd.grad` 核对。能解释这个结果，并顺着 Laplace 的训练循环指出“坐标在哪、参数在哪、损失是什么、谁被更新”，就有基础继续读本仓库了。

直接运行已有短程入口（在 PINN 项目目录）：

```bash
python psi-NN/repro/run_mac.py --case Laplace --device cpu --steps 100 --tag learning
```

这只训练 100 步用于看懂流程。直接运行原来的 Panel.py 则会启动多个长程实验，阅读阶段先使用短程入口。
