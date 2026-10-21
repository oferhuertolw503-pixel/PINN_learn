"""Check architectural claims at random initialization, without fitting any data."""
from pathlib import Path
import importlib
import json
import os
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get('REPO_ROOT', str(ROOT / 'psi-NN' / 'Psi-NN'))).resolve()
if not (UPSTREAM / 'Module').is_dir():
    raise SystemExit('Missing upstream checkout; follow the root README or set REPO_ROOT')
sys.path.insert(0, str(UPSTREAM))
torch.manual_seed(1234)
xy = torch.rand(128, 2) * 2 - 1
flip = xy.clone()
flip[:, 1] *= -1
checks = {}
with torch.no_grad():
    laplace = importlib.import_module('Module.PsiNN_laplace').Net(8)
    poisson = importlib.import_module('Module.PsiNN_poisson').Net(20)
    burgers = importlib.import_module('Module.PsiNN_burgers').Net(10)
    flow = importlib.import_module('Module.PsiNN_flow').Net(20, 3)
    checks['laplace_even_y_max_error'] = (laplace(xy)-laplace(flip)).abs().max().item()
    checks['poisson_swap_max_error'] = (poisson(xy)-poisson(xy[:, [1, 0]])).abs().max().item()
    checks['burgers_odd_y_max_error'] = (burgers(xy)+burgers(flip)).abs().max().item()
    checks['burgers_output_bias'] = burgers.fc4_1.bias.item()
    checks['burgers_odd_error_minus_twice_bias'] = (burgers(xy)+burgers(flip)-2*burgers.fc4_1.bias).abs().max().item()
    y, yf = flow(xy), flow(flip)
    checks['flow_even_p_u_max_error'] = (y[:, :2]-yf[:, :2]).abs().max().item()
    checks['flow_odd_v_max_error'] = (y[:, 2]+yf[:, 2]).abs().max().item()
checks['note'] = 'Untrained networks. Nonzero errors show a property is not guaranteed for all parameter values.'
output = UPSTREAM / 'Results' / 'structure_checks.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
