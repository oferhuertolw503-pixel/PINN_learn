"""Retrain fixed symbol-matrix relations extracted from a Burgers or Laplace student.

This is a cluster-sharing reconstruction of the student topology.  It does
not claim to reproduce the paper's additional hand-designed block merging.
"""
import argparse
import os
from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1] / 'Psi-NN'
sys.path.insert(0, str(REPO_ROOT))
from Module import Training  # noqa: E402
from Module.PINN import Net  # noqa: E402
from clustered_burgers import ClusteredBurgers, ClusteredLinear  # noqa: E402

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']


def verify_relation_network(structure_path, student_checkpoint):
    constrained = ClusteredBurgers(structure_path, init='centers').eval()
    layers = [next(layer for layer in constrained.layers if isinstance(layer, ClusteredLinear)).in_features]
    layers += [layer.out_features for layer in constrained.layers if isinstance(layer, ClusteredLinear)]
    dense = Net(layers).eval()
    student = Net(layers).eval()
    student.load_state_dict(torch.load(student_checkpoint, map_location='cpu', weights_only=True))
    with torch.no_grad():
        for i, layer in enumerate(layer for layer in constrained.layers if isinstance(layer, ClusteredLinear)):
            target = dense.layers._modules[f'layer_{i}']
            target.weight.copy_(layer.weight_centers[layer.weight_index] * layer.weight_sign)
            target.bias.copy_(layer.bias_centers[layer.bias_index] * layer.bias_sign)
        points = torch.rand(1000, 2)
        points[:, 1] = 2 * points[:, 1] - 1
        max_diff = (constrained(points) - dense(points)).abs().max().item()
        student_mse = torch.mean((constrained(points) - student(points)) ** 2).item()
    print(f'关系网络参数: {sum(p.numel() for p in constrained.parameters())} '
          f'/ 原学生参数: {constrained.dense_parameter_count}')
    print(f'逐点重建最大偏差: {max_diff:.3e}')
    print(f'簇中心替换相对原学生的随机点 MSE: {student_mse:.3e}')
    if max_diff > 1e-6:
        raise SystemExit('关系网络验证失败')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', choices=('Burgers_inv_distill', 'Laplace_distill'),
                        default='Burgers_inv_distill')
    parser.add_argument('--ini', default='PAPER')
    parser.add_argument('--linkage', choices=('ward', 'average'), default='average')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='Run PDE/data optimization steps without final outputs')
    parser.add_argument('--smoke-steps', type=int, default=1)
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    cluster_dir = 'Cluster' if args.linkage == 'ward' else 'Cluster_average'
    prefix = f'{args.question}_{args.ini}'
    structure_path = (REPO_ROOT / 'Results' / prefix /
                      cluster_dir / f'{prefix}_structure.json')
    if not structure_path.exists():
        raise SystemExit(f'找不到结构: {structure_path}')
    student_checkpoint = (REPO_ROOT / 'Results' / prefix / 'Models' /
                          f'{prefix}_PINN_student.pth')
    torch.manual_seed(1234)
    verify_relation_network(structure_path, student_checkpoint)
    if args.verify_only:
        return

    experiment = Training.model(args.question, args.ini)
    # Stage 3 uses the teacher's PDE + data loss, without a second distillation.
    experiment.distill_state = False
    experiment.save_desti = f'./Results/{prefix}_reconstructed_{args.linkage}/'
    experiment.original_lr = experiment.learning_rate
    experiment.net = ClusteredBurgers(structure_path, init='random').float().to(Training.device)
    experiment.mesh_init()
    if args.smoke_test:
        experiment.train_steps = args.smoke_steps
        experiment.pace_record_state = 0
    experiment.train_adam()
    if args.smoke_test:
        print(f'冒烟训练末步损失: {experiment.loss.item():.6g}')
        return
    experiment.model_save()
    experiment.result_show()


if __name__ == '__main__':
    main()
