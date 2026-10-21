# coding = utf-8
"""Psi-NN 最小复现 —— 结构提取（第三步）

把 pic_parameter.ipynb 的第 2、4 个代码单元搬成无头脚本：
对训练好的**学生网络**逐层取出 weight / bias，按绝对值做层次聚类
（默认 Ward，与 notebook 一致；可选 average，与论文补充材料一致；阈值默认 0.1），
再把每个权重替换成「所属聚类中心 × 原符号」，
输出 A / B / -A / -B 这样的字母矩阵 —— 这就是论文里「自动发现 PDE 结构」的那一步。

用法（工作目录需为 Psi-NN 代码根目录）：
    python extract_structure.py
    python extract_structure.py --ckpt Results/xxx/Models/yyy.pth --threshold 0.1
"""
import argparse
import json
import os
import string

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
# notebook 里只写了 Times New Roman；Linux 上一般没这个字体，
# 这里补一个 DejaVu Sans 兜底，免得每个文字都刷一遍 findfont 警告。
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.linewidth'] = 1


def cluster_by_abs(abs_weights, threshold, method='ward'):
    """按 |w| 做层次聚类；Ward 与作者 notebook 一致，average 与论文补充材料一致。

    返回 (labels, centers)：labels 是 1-based 簇编号，centers 是各簇 |w| 均值。
    """
    if len(abs_weights) < 2:
        return np.ones(len(abs_weights), dtype=int), [float(abs_weights.mean())]

    labels = fcluster(linkage(abs_weights, method=method), threshold, criterion='distance')
    centers = [float(abs_weights[labels == lab].mean()) for lab in np.unique(labels)]
    return labels, centers


def main():
    ap = argparse.ArgumentParser(description='Psi-NN 学生网络结构提取')
    ap.add_argument('--ques', default='Burgers_inv_distill', help='算例名')
    ap.add_argument('--ini', default='REPRO', help='配置号')
    ap.add_argument('--ckpt', default=None, help='学生网络 .pth 路径（默认取该算例的最终学生权重）')
    ap.add_argument('--outdir', default=None, help='输出目录（默认 Results/<算例>_<配置号>/Cluster）')
    ap.add_argument('--threshold', type=float, default=0.1, help='聚类距离阈值，notebook 里是 0.1')
    ap.add_argument('--linkage', choices=('ward', 'average'), default='ward',
                    help='簇间距离：ward 对应作者 notebook，average 对应论文补充材料')
    args = ap.parse_args()

    ckpt = args.ckpt or f'./Results/{args.ques}_{args.ini}/Models/{args.ques}_{args.ini}_PINN_student.pth'
    if not os.path.isfile(ckpt):
        raise SystemExit(
            f'找不到权重文件：{os.path.abspath(ckpt)}\n'
            '先跑 02_run_repro.sh 完成训练，或用 --ckpt 指定已有权重。'
        )

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.dirname(ckpt)), 'Cluster')
    os.makedirs(outdir, exist_ok=True)

    state = torch.load(ckpt, map_location='cpu')
    print(f'权重文件: {os.path.abspath(ckpt)}')
    print(f'层      : {list(state.keys())}')
    print(f'阈值    : {args.threshold}')
    print(f'簇间距离: {args.linkage}')
    print()

    rng = np.random.default_rng(1234)  # notebook 用无种子的 np.random.uniform，这里固定住以便复现
    structure = {}

    for name, param in state.items():
        layer_weights = param.detach().cpu().numpy()
        flat = layer_weights.flatten()
        abs_w = np.abs(flat).reshape(-1, 1)

        labels, centers = cluster_by_abs(abs_w, args.threshold, args.linkage)

        letters = {i + 1: ch for i, ch in enumerate(string.ascii_uppercase[:len(centers)])}
        mapping = {ch: c for ch, c in zip(letters.values(), centers)}

        symbols = np.array([
            f"{'-' if w < 0 else ''}{letters[lab]}" if lab in letters else '0'
            for w, lab in zip(flat, labels)
        ]).reshape(layer_weights.shape)

        # 分布图：正值蓝、负值取绝对值后红、绿虚线为聚类中心
        pos = flat[flat >= 0]
        neg = -flat[flat < 0]
        plt.figure(figsize=(4, 1.5))
        plt.scatter(pos, rng.uniform(-0.1, 0.1, size=len(pos)),
                    color='blue', alpha=0.5, s=15, label=f'Positive {name}')
        plt.scatter(neg, rng.uniform(-0.1, 0.1, size=len(neg)),
                    color='red', alpha=0.5, s=15, label=f'Negative {name} (converted to positive)')
        for i, c in enumerate(centers):
            plt.axvline(x=c, color='green', linestyle='--', label='Cluster Center' if i == 0 else '')
        plt.yticks([])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'{args.ques}_{args.ini}_{name}_distribution.png'), dpi=300)
        plt.close()

        structure[name] = {
            'shape': list(layer_weights.shape),
            'n_clusters': len(centers),
            'centers': mapping,
            'symbols': symbols.tolist(),
        }

        print(f'Layer: {name}   shape={layer_weights.shape}   聚类数={len(centers)}')
        print('Symbolized Weights Matrix:')
        print(symbols)
        print('Cluster Mapping (Letter -> Center):')
        for ch, c in mapping.items():
            print(f'{ch}: {c}')
        print()

    json_path = os.path.join(outdir, f'{args.ques}_{args.ini}_structure.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)

    print(f'分布图     -> {outdir}')
    print(f'结构化结果 -> {json_path}')


if __name__ == '__main__':
    main()
