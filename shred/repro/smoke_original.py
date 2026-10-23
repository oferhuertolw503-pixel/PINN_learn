"""用原作者 SHRED 模型走通小型解析热方程数据的重建链路；不是论文复现。"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = 'e5b122259f59cb0d4a91fa60da812349616380b3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use CPU or check the existing WSL environment')
    upstream = ROOT / 'shred' / 'upstream' / 'original-shred'
    if not (upstream / 'models.py').is_file():
        parser.error('Missing original-shred checkout; see shred/README.md')
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != EXPECTED_COMMIT:
        parser.error('Upstream commit differs from materials.json; inspect before running')
    # 直接导入下载的作者类，不复制/重写它的 LSTM 与解码器。
    spec = importlib.util.spec_from_file_location('original_shred_models', upstream / 'models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(1234)
    torch.set_num_threads(4)
    rng = np.random.default_rng(1234)

    # 生成周期域上的解析热方程解：u_t=kappa*u_xx。
    # 每条轨迹的幅值、相位和扩散系数不同；网络训练只用全场标签，不使用方程残差。
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    t = np.linspace(0, 2, 64)
    fields = []
    for _ in range(32):
        kappa = rng.uniform(0.05, 0.2)
        a, b = rng.uniform(0.5, 1.5, 2)
        p, q = rng.uniform(0, 2 * np.pi, 2)
        u = a * np.exp(-kappa * t[:, None]) * np.sin(x[None, :] + p)
        u += b * np.exp(-4 * kappa * t[:, None]) * np.cos(2 * x[None, :] + q)
        fields.append(u)
    fields = torch.tensor(np.stack(fields), dtype=torch.float32)
    # [轨迹,时间,全场坐标]=[32,64,64]。按完整轨迹划分，避免同条轨迹窗口混进测试集。
    train_ids = list(range(24))
    valid_ids = list(range(24, 28))
    test_ids = list(range(28, 32))
    mean = fields[train_ids].mean(dim=(0, 1))
    std = fields[train_ids].std(dim=(0, 1)).clamp_min(1e-6)
    normalized = (fields - mean) / std  # 只用训练集拟合归一化统计量。
    sensors, lag = [3, 22, 43], 8

    def windows(ids):
        data = normalized[ids]
        # 每个输入为连续8步×3个传感器，目标为窗口最后时刻的64点全场。
        seq = data[:, :, sensors].unfold(1, lag, 1).permute(0, 1, 3, 2)
        inputs = seq.reshape(-1, lag, len(sensors)).to(args.device)
        targets = data[:, lag - 1:].reshape(-1, len(x)).to(args.device)
        return inputs, targets

    train_x, train_y = windows(train_ids)
    valid_x, valid_y = windows(valid_ids)
    test_x, test_y = windows(test_ids)
    model = module.SHRED(3, 64, hidden_size=16, hidden_layers=1, l1=32, l2=32).to(args.device)
    # LSTM 将 [batch,8,3] 变成每样本16维隐藏状态，MLP 再输出 [batch,64]。
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    best, best_state, history = float('inf'), None, []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for sensor_window, target_field in loader:
            prediction = model(sensor_window)
            loss = ((prediction - target_field)**2).mean()
            optimizer.zero_grad()
            loss.backward()  # 这里仅需对网络权重反传，不像 PINN 那样对坐标求导。
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = ((model(valid_x) - valid_y)**2).mean().item()
        history.append(dict(epoch=epoch, validation_normalized_mse=val))
        if val < best:
            best, best_state = val, deepcopy(model.state_dict())
    model.load_state_dict(best_state)  # 只用验证集选模型，测试集留到最后一次评分。
    model.eval()
    with torch.no_grad():
        pred = model(test_x).cpu() * std + mean
        truth = test_y.cpu() * std + mean
        mse = ((pred - truth)**2).mean().item()
        rel = (torch.linalg.vector_norm(pred - truth) / torch.linalg.vector_norm(truth)).item()
        baseline = ((mean - truth)**2).mean().item()
    summary = dict(
        status='synthetic pipeline smoke only; not SST, PDEBench or a paper reproduction',
        seed=1234, original_shred_commit=commit, device=args.device, torch=torch.__version__,
        epochs=args.epochs, data_shape=list(fields.shape), sensor_indices=sensors, lag=lag,
        split_trajectory_ids=dict(train=train_ids, validation=valid_ids, test=test_ids),
        input_shape=list(train_x.shape), target_shape=list(train_y.shape),
        hidden_size=16, hidden_layers=1, decoder_sizes=[32, 32],
        parameters=sum(p.numel() for p in model.parameters()),
        test_physical_mse=mse, test_relative_l2=rel, training_mean_baseline_mse=baseline,
        elapsed_seconds=time.perf_counter() - start, history=history,
    )
    output = ROOT / 'shred' / 'Results' / 'synthetic_heat_smoke.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('Saved:', output)


if __name__ == '__main__':
    main()
