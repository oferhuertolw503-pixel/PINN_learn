# coding = utf-8
"""Psi-NN 最小复现 —— 训练入口。

等价于 Panel.py 里的 `Training.model(...).train()`，但只跑一个算例，
并且显式把代码根目录塞进 sys.path（本脚本放于 repro/ 下并按绝对路径调用，
Python 只会把脚本自身所在目录加进 sys.path，不会加 cwd）。

用法：
    python run_train.py [算例名] [配置号]      # 默认 Burgers_inv_distill REPRO
"""
import os
import sys

REPO_ROOT = os.environ.get('REPO_ROOT') or os.getcwd()
if not os.path.isdir(os.path.join(REPO_ROOT, 'Module')):
    raise SystemExit(
        f'在 {REPO_ROOT} 下找不到 Module/ 目录。\n'
        '请在 Psi-NN 代码根目录下运行，或设置 REPO_ROOT 环境变量。'
    )
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch  # noqa: E402
import Module.Training as Training  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

torch.manual_seed(1234)  # 与 Panel.py 一致

QUES = sys.argv[1] if len(sys.argv) > 1 else 'Burgers_inv_distill'
INI = sys.argv[2] if len(sys.argv) > 2 else 'REPRO'

model = Training.model(QUES, INI)

print(f'=== 训练 {QUES}_{INI} ===')
print(f'device      : {Training.device}')
print(f'教师网络    : {model.layer}')
print(f'教师步数    : {model.train_steps}')
if model.distill_state:
    print(f'学生网络    : {model.layer_student}')
    print(f'学生步数    : {int(model.train_steps * model.train_ratio)}')
print()

model.train()
