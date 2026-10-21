"""Independent full-field check for the Burgers inverse experiment.

Coordinates in the repository are (time, space).  The Cole--Hopf heat
integral supplies the reference for u_t + u*u_x - nu*u_xx = 0 with
u(0, x)=-sin(pi*x), u(t, +/-1)=0, and nu=0.01/pi.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.special import roots_hermite


REPO_ROOT = Path(__file__).resolve().parents[1] / "Psi-NN"
sys.path.insert(0, str(REPO_ROOT))
from Module.PINN import Net  # noqa: E402
from Module.PINN_post_minus import Net as PostNet  # noqa: E402
from Module.PsiNN_burgers import Net as AuthorPsiNet  # noqa: E402
from clustered_burgers import ClusteredBurgers  # noqa: E402


NU_TRUE = 0.01 / math.pi


def exact_burgers(coords, quadrature_order=192, batch_size=4096):
    """Cole--Hopf expectation; the initial potential is 2-periodic.

    Its periodic extension enforces the homogeneous Dirichlet boundaries
    for u at x=+/-1.  Evaluation is batched to limit host memory.
    """
    z, weights = roots_hermite(quadrature_order)
    a = 1.0 / (2.0 * NU_TRUE * math.pi)
    result = np.empty(len(coords), dtype=np.float64)
    for start in range(0, len(coords), batch_size):
        block = coords[start:start + batch_size]
        tau = block[:, 0:1]
        x = block[:, 1:2]
        shifted = x + np.sqrt(4.0 * NU_TRUE * tau) * z
        phase = math.pi * shifted
        potential = np.exp(-a * np.cos(phase))
        weighted = potential * weights
        result[start:start + batch_size] = -np.sum(weighted * np.sin(phase), axis=1) / np.sum(weighted, axis=1)
    return result


def predict(network, coords, device):
    network = network.to(device)
    network.eval()
    predictions = []
    with torch.no_grad():
        for block in np.array_split(coords, max(1, math.ceil(len(coords) / 4096))):
            predictions.append(network(torch.as_tensor(block, dtype=torch.float32, device=device)).cpu().numpy().ravel())
    return np.concatenate(predictions)


def errors(prediction, truth, grid):
    residual = prediction - truth
    field = prediction.reshape(grid, grid)  # rows are space, columns are time
    space = np.linspace(-1, 1, grid)
    return {
        "mse": float(np.mean(residual ** 2)),
        "relative_l2": float(np.linalg.norm(residual) / np.linalg.norm(truth)),
        "max_abs": float(np.max(np.abs(residual))),
        "odd_symmetry_mse": float(np.mean((field + field[::-1, :]) ** 2)),
        "initial_mse": float(np.mean((field[:, 0] + np.sin(math.pi * space)) ** 2)),
        "boundary_mse": float(np.mean(np.vstack((field[0], field[-1])) ** 2)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="Burgers_inv_distill")
    parser.add_argument("--ini", default="PAPER")
    parser.add_argument("--step", type=int, default=None, help="Use an intermediate checkpoint")
    parser.add_argument("--grid", type=int, default=200)
    parser.add_argument("--quadrature", type=int, default=192)
    parser.add_argument("--linkage", choices=("ward", "average"), default="average")
    parser.add_argument("--output", default=None, help="Optional JSON metrics path")
    args = parser.parse_args()

    t, x = np.meshgrid(np.linspace(0, 1, args.grid), np.linspace(-1, 1, args.grid))
    coords = np.column_stack((t.ravel(), x.ravel()))
    truth = exact_burgers(coords, quadrature_order=args.quadrature)
    # Check that the analytic convention agrees with the data shipped by the authors.
    observations = np.vstack([
        np.loadtxt(REPO_ROOT / "Database" / f"Burgers_inv_data_{serial}.csv", delimiter=",")
        for serial in (1, 3)
    ])
    observed_error = exact_burgers(observations[:, :2], quadrature_order=args.quadrature) - observations[:, 2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prefix = f"{args.question}_{args.ini}"
    model_dir = REPO_ROOT / "Results" / prefix / "Models"
    suffix = "" if args.step is None else f"_step_{args.step}"
    metrics = {
        "nu_true": NU_TRUE,
        "experiment": prefix,
        "evaluation_domain": {"time": [0, 1], "space": [-1, 1]},
        "grid": args.grid,
        "quadrature_order": args.quadrature,
        "sparse_data_count": len(observations),
        "sparse_data_reference_mse": float(np.mean(observed_error ** 2)),
        "sparse_data_reference_max_abs": float(np.max(np.abs(observed_error))),
        "models": {},
    }
    if args.question == "Burgers_inv_distill":
        candidates = (("teacher", "PINN", Net), ("student", "PINN_student", Net))
    elif args.question == "Burgers_inv":
        candidates = (("PINN", "PINN", Net),
                      ("PINN_post_minus", "PINN_post_minus", PostNet),
                      ("PsiNN_burgers", "PsiNN_burgers", AuthorPsiNet))
    else:
        raise SystemExit(f"Unsupported question: {args.question}")
    for role, model_name, network_class in candidates:
        name = f"{prefix}_{model_name}{suffix}.pth"
        checkpoint = model_dir / name
        if not checkpoint.exists():
            continue
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if role == "PsiNN_burgers":
            width = state['fc1_1.weight'].shape[0]
            network = AuthorPsiNet(width)
            architecture = {"submatrix_dimension": width}
        else:
            weight_shapes = [value.shape for key, value in state.items() if key.endswith('.weight')]
            layers = [weight_shapes[0][1]] + [shape[0] for shape in weight_shapes]
            network = network_class(layers)
            architecture = {"layers": layers}
        network.load_state_dict(state)
        prediction = predict(network, coords, device)
        metrics["models"][role] = {
            "checkpoint": str(checkpoint),
            **architecture,
            "trainable_parameters": sum(p.numel() for p in network.parameters()),
            **errors(prediction, truth, args.grid),
        }

    if args.question == "Burgers_inv_distill" and args.step is None:
        cluster_dir = "Cluster" if args.linkage == "ward" else "Cluster_average"
        structure_path = REPO_ROOT / "Results" / f"Burgers_inv_distill_{args.ini}" / cluster_dir / f"Burgers_inv_distill_{args.ini}_structure.json"
        if structure_path.exists():
            for role, checkpoint in (
                ("cluster_centers", None),
                ("reconstructed", REPO_ROOT / "Results" / f"Burgers_inv_distill_{args.ini}_reconstructed_{args.linkage}" / "Models" / f"Burgers_inv_distill_{args.ini}_clustered_burgers.pth"),
            ):
                if checkpoint is not None and not checkpoint.exists():
                    continue
                network = ClusteredBurgers(structure_path, init='centers')
                if checkpoint is not None:
                    network.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
                prediction = predict(network, coords, device)
                metrics["models"][role] = {
                    "checkpoint": str(checkpoint or structure_path),
                    "trainable_parameters": sum(p.numel() for p in network.parameters()),
                    **errors(prediction, truth, args.grid),
                }

    output = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
