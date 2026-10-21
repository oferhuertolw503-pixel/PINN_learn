"""Compare the first 1000 optimized teacher steps with the saved unmodified run."""
import csv
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Psi-NN'))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch
from Module import PINN, Training

torch.manual_seed(1234)
model = Training.model('Burgers_inv_distill', 'REPRO')
model.net = PINN.Net(model.layer).float().to(Training.device)
model.net_student = PINN.Net(model.layer_student).float().to(Training.device)
model.mesh_init()
model.train_steps = 1000
model.train_ratio = 0
model.pace_record_state = 0
started = time.monotonic()
model.train_adam()

reference_path = os.path.join(ROOT, 'Results', 'Burgers_inv_distill_REPRO',
                              'Models', 'Burgers_inv_distill_REPRO_PINN_step_1000.pth')
reference = torch.load(reference_path, map_location=Training.device, weights_only=True)
weight_delta = max((tensor - reference[name]).abs().max().item()
                   for name, tensor in model.net.state_dict().items())
param_path = os.path.join(ROOT, 'Results', 'Burgers_inv_distill_REPRO',
                          'Parameters', 'Burgers_inv_distill_REPRO_paras_PINN.csv')
with open(param_path, newline='') as stream:
    recorded_parameter = float(list(csv.DictReader(stream))[999]['parameters_1'])
parameter_delta = abs(model.para_undetermin.item() - recorded_parameter)
print(f'1000-step weight max absolute difference: {weight_delta:.9g}')
print(f'1000-step parameter absolute difference: {parameter_delta:.9g}')
print(f'elapsed seconds: {time.monotonic() - started:.3f}')
assert weight_delta < 1e-5 and parameter_delta < 1e-5
