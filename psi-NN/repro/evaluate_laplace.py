"""Evaluate the paper's Laplace case against u(x, y) = x**3 - 3*x*y**2."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1] / "Psi-NN"
sys.path.insert(0, str(REPO_ROOT))
from Module.PINN import Net  # noqa: E402
from Module.PINN_post_plus import Net as PostNet  # noqa: E402
from Module.PsiNN_laplace import Net as AuthorPsiNet  # noqa: E402
from clustered_burgers import ClusteredBurgers  # noqa: E402


def predict(network, coords, device):
    network = network.to(device).eval()
    values = []
    with torch.no_grad():
        for block in np.array_split(coords, math.ceil(len(coords) / 4096)):
            values.append(network(torch.as_tensor(block, dtype=torch.float32, device=device)).cpu().numpy().ravel())
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", choices=("Laplace", "Laplace_distill"), default="Laplace_distill")
    parser.add_argument("--ini", default="PAPER")
    parser.add_argument("--grid", type=int, default=200)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--structure", default=None,
                        help="Optional student cluster JSON; evaluate center replacement without retraining")
    args = parser.parse_args()

    axis = np.linspace(-1, 1, args.grid)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    coords = np.column_stack((x.ravel(), y.ravel()))
    truth = x**3 - 3 * x * y**2
    prefix = f"{args.question}_{args.ini}"
    directory = REPO_ROOT / "Results" / prefix / "Models"
    suffix = "" if args.step is None else f"_step_{args.step}"
    candidates = (
        (("teacher", "PINN", Net), ("student", "PINN_student", Net))
        if args.question == "Laplace_distill" else
        (("PINN", "PINN", Net), ("PINN_post_plus", "PINN_post_plus", PostNet),
         ("PsiNN_laplace", "PsiNN_laplace", AuthorPsiNet))
    )
    result = {"experiment": prefix, "reference": "x**3 - 3*x*y**2", "grid": args.grid, "models": {}}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for role, name, network_class in candidates:
        checkpoint = directory / f"{prefix}_{name}{suffix}.pth"
        if not checkpoint.exists():
            continue
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if role == "PsiNN_laplace":
            network = network_class(state["fc1_2.weight"].shape[0])
        else:
            weights = [value.shape for key, value in state.items() if key.endswith(".weight")]
            layers = [weights[0][1]] + [shape[0] for shape in weights]
            network = network_class(layers)
        network.load_state_dict(state)
        field = predict(network, coords, device).reshape(args.grid, args.grid)
        residual = field - truth
        boundary = np.concatenate((residual[0], residual[-1], residual[:, 0], residual[:, -1]))
        result["models"][role] = {
            "checkpoint": str(checkpoint),
            "trainable_parameters": sum(p.numel() for p in network.parameters()),
            "mse": float(np.mean(residual**2)),
            "relative_l2": float(np.linalg.norm(residual) / np.linalg.norm(truth)),
            "max_abs": float(np.max(np.abs(residual))),
            "even_symmetry_mse": float(np.mean((field - field[:, ::-1])**2)),
            "boundary_mse": float(np.mean(boundary**2)),
        }

    if args.structure:
        if args.question != "Laplace_distill":
            parser.error("--structure requires --question Laplace_distill")
        network = ClusteredBurgers(args.structure, init="centers")
        field = predict(network, coords, device).reshape(args.grid, args.grid)
        residual = field - truth
        boundary = np.concatenate((residual[0], residual[-1], residual[:, 0], residual[:, -1]))
        result["models"]["cluster_replacement"] = {
            "structure": str(Path(args.structure).resolve()),
            "trainable_parameters_if_retrained": sum(p.numel() for p in network.parameters()),
            "dense_parameter_count": network.dense_parameter_count,
            "mse": float(np.mean(residual**2)),
            "relative_l2": float(np.linalg.norm(residual) / np.linalg.norm(truth)),
            "max_abs": float(np.max(np.abs(residual))),
            "even_symmetry_mse": float(np.mean((field - field[:, ::-1])**2)),
            "boundary_mse": float(np.mean(boundary**2)),
        }

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
