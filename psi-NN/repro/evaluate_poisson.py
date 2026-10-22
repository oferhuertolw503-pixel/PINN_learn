"""Independently evaluate the upstream Poisson models against their trained PDE.

Training.py minimizes Delta u - f, where
f = 1/2 sum_{k=1}^4 (-1)^(k+1) k^2 sin(k*pi*x) sin(k*pi*y),
with zero boundary values. Its net_global() uses a different reference field;
both errors are reported so the discrepancy remains visible.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1] / "Psi-NN"
sys.path.insert(0, str(REPO_ROOT))
from Module.PINN import Net as PINN  # noqa: E402
from Module.PINN_post_poisson import Net as PostNet  # noqa: E402
from Module.PsiNN_poisson import Net as AuthorPsiNet  # noqa: E402


def reference_fields(coords):
    k = np.arange(1, 5, dtype=np.float64)[:, None]
    basis = np.sin(k * math.pi * coords[:, 0]) * np.sin(k * math.pi * coords[:, 1])
    signs = (-1.0) ** (k + 1)
    # Delta(sin(k*pi*x) sin(k*pi*y)) = -2*k^2*pi^2 times the same basis.
    pde_consistent = -np.sum(signs * basis, axis=0) / (4 * math.pi**2)
    legacy_net_global = np.sum(signs * k * basis, axis=0) / (4 * math.pi**2)
    return pde_consistent, legacy_net_global


def predict(network, coords, device):
    network = network.to(device).eval()
    blocks = []
    with torch.no_grad():
        for start in range(0, len(coords), 4096):
            block = torch.as_tensor(coords[start:start + 4096], dtype=torch.float32, device=device)
            blocks.append(network(block).cpu().numpy().ravel())
    return np.concatenate(blocks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ini", default="REPRO")
    parser.add_argument("--grid", type=int, default=200)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    axis = np.linspace(0, 1, args.grid)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    coords = np.column_stack((x.ravel(), y.ravel()))
    truth, legacy = reference_fields(coords)
    prefix = f"Poisson_{args.ini}"
    model_dir = REPO_ROOT / "Results" / prefix / "Models"
    suffix = "" if args.step is None else f"_step_{args.step}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {
        "experiment": prefix,
        "checkpoint_step": args.step if args.step is not None else "final",
        "grid": args.grid,
        "pde": "Delta u = 0.5 sum((-1)^(k+1) k^2 sin(k pi x) sin(k pi y)), k=1..4",
        "reference": "-sum((-1)^(k+1) sin(k pi x) sin(k pi y))/(4 pi^2)",
        "legacy_reference_conflict_mse": float(np.mean((truth - legacy) ** 2)),
        "models": {},
    }
    candidates = (("PINN", PINN), ("PINN_post_poisson", PostNet), ("PsiNN_poisson", AuthorPsiNet))
    for name, network_class in candidates:
        checkpoint = model_dir / f"{prefix}_{name}{suffix}.pth"
        if not checkpoint.exists():
            continue
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if name == "PsiNN_poisson":
            network = network_class(state["fc1.weight"].shape[0] // 2)
        else:
            weights = [value.shape for key, value in state.items() if key.endswith(".weight")]
            layers = [weights[0][1]] + [shape[0] for shape in weights]
            network = network_class(layers)
        network.load_state_dict(state)
        field = predict(network, coords, device).reshape(args.grid, args.grid)
        residual = field - truth.reshape(args.grid, args.grid)
        boundary = np.concatenate((residual[0], residual[-1], residual[:, 0], residual[:, -1]))
        result["models"][name] = {
            "checkpoint": str(checkpoint),
            "trainable_parameters": sum(p.numel() for p in network.parameters()),
            "pde_consistent_mse": float(np.mean(residual**2)),
            "pde_consistent_relative_l2": float(np.linalg.norm(residual) / np.linalg.norm(truth)),
            "legacy_net_global_mse": float(np.mean((field.ravel() - legacy) ** 2)),
            "boundary_mse": float(np.mean(boundary**2)),
            "exchange_symmetry_mse": float(np.mean((field - field.T) ** 2)),
        }

    if not result["models"]:
        raise SystemExit(f"No checkpoints found in {model_dir}")
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
