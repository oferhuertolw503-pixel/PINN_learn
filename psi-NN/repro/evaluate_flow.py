"""Compare Flow checkpoints with the field CSV shipped in the author repository.

The CSV's provenance is not established. Its coordinates are used for PDE
collocation, but its interior p/u/v labels are not directly used in the loss.
This is a transductive, in-domain comparison, not unseen-coordinate or
experimental validation.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1] / "Psi-NN"
sys.path.insert(0, str(REPO_ROOT))
from Module.PINN import Net as PINN  # noqa: E402
from Module.PINN_post_flow import Net as PostNet  # noqa: E402
from Module.PsiNN_flow import Net as AuthorPsiNet  # noqa: E402


def predict(network, coords, device):
    network = network.to(device).eval()
    blocks = []
    with torch.no_grad():
        for start in range(0, len(coords), 4096):
            block = torch.as_tensor(coords[start:start + 4096], dtype=torch.float32, device=device)
            blocks.append(network(block).cpu().numpy())
    return np.concatenate(blocks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ini", default="PILOT")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data = pd.read_csv(REPO_ROOT / "Database" / "flow" / "fluid_data.csv")
    coords = data[["x", "y"]].to_numpy(dtype=np.float64)
    coords[:, 1] -= 0.2  # Match Training.mesh_init().
    truth = data[["p", "u", "v"]].to_numpy(dtype=np.float64)
    prefix = f"Flow_{args.ini}"
    model_dir = REPO_ROOT / "Results" / prefix / "Models"
    suffix = "" if args.step is None else f"_step_{args.step}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {
        "experiment": prefix,
        "checkpoint_step": args.step if args.step is not None else "final",
        "reference": "author repository Database/flow/fluid_data.csv; provenance unverified",
        "comparison": "transductive: coordinates used for PDE collocation; interior field labels not directly used in loss",
        "expected_y_parity": "p and u even; v odd about the shifted y=0 axis",
        "rows": len(data),
        "models": {},
    }
    candidates = (("PINN", PINN), ("PINN_post_flow", PostNet), ("PsiNN_flow", AuthorPsiNet))
    for name, network_class in candidates:
        checkpoint = model_dir / f"{prefix}_{name}{suffix}.pth"
        if not checkpoint.exists():
            continue
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if name == "PsiNN_flow":
            network = network_class(state["fc1_1.weight"].shape[0], output_num=3)
        else:
            weights = [value.shape for key, value in state.items() if key.endswith(".weight")]
            layers = [weights[0][1]] + [shape[0] for shape in weights]
            network = network_class(layers)
        network.load_state_dict(state)
        prediction = predict(network, coords, device)
        reflected_coords = coords.copy()
        reflected_coords[:, 1] *= -1
        reflected = predict(network, reflected_coords, device)
        errors = prediction - truth
        metrics = {}
        for column, label in enumerate(("p", "u", "v")):
            symmetry_error = prediction[:, column] - reflected[:, column]
            if label == "v":
                symmetry_error = prediction[:, column] + reflected[:, column]
            metrics[label] = {
                "mse": float(np.mean(errors[:, column] ** 2)),
                "relative_l2": float(np.linalg.norm(errors[:, column]) / np.linalg.norm(truth[:, column])),
                "expected_y_parity_mse": float(np.mean(symmetry_error**2)),
            }
        pressure_offset = float(np.mean(errors[:, 0]))
        result["models"][name] = {
            "checkpoint": str(checkpoint),
            "trainable_parameters": sum(p.numel() for p in network.parameters()),
            "components": metrics,
            "pressure_mean_error": pressure_offset,
            "pressure_gauge_aligned_mse": float(np.mean((errors[:, 0] - pressure_offset) ** 2)),
        }

    if not result["models"]:
        raise SystemExit(f"No checkpoints found in {model_dir}")
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
