"""Run one upstream experiment at a time without editing the downloaded source."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get('REPO_ROOT', str(ROOT / 'psi-NN' / 'Psi-NN'))).resolve()
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(UPSTREAM / 'Results' / '.matplotlib'))
sys.path.insert(0, str(UPSTREAM))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', default='Laplace', choices=['Laplace', 'Burgers_inv', 'Burgers_inv_distill', 'Poisson', 'Flow'])
    p.add_argument('--device', default='cpu', choices=['cpu', 'mps'])
    p.add_argument('--steps', type=int, default=100, help='Short-run iterations (default: 100); use an explicit count for longer runs')
    p.add_argument('--grid', type=int, help='Override interior grid side length (Flow uses supplied mesh)')
    p.add_argument('--models', nargs='+', help='Subset of model names in the selected upstream config')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--tag', default='mac')
    args = p.parse_args()
    if args.steps is not None and args.steps < 1:
        p.error('--steps must be positive')
    if args.grid is not None and args.grid < 2:
        p.error('--grid must be at least 2')
    if not (UPSTREAM / 'Module' / 'Training.py').is_file():
        p.error('Missing upstream checkout; follow the root README or set REPO_ROOT')
    if args.threads < 1:
        p.error('--threads must be positive')
    import torch
    import Module.Training as training
    torch.set_num_threads(args.threads)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available in this process')
    training.device = torch.device(args.device)
    os.chdir(UPSTREAM)
    base = training.model(args.case, 'EXP')
    models = args.models or base.model_ini_dict['model']
    if not set(models) <= set(base.model_ini_dict['model']):
        p.error('Requested model is not in the upstream config')
    output = UPSTREAM / 'Results' / 'mac_smoke' / args.tag / f'{args.case}_{args.device}'
    output.mkdir(parents=True, exist_ok=True)
    results = []
    metadata = dict(arguments=vars(args), python=platform.python_version(), torch=torch.__version__,
                    mps_built=torch.backends.mps.is_built(), mps_available=torch.backends.mps.is_available(),
                    note='Short runs test execution and cost, not paper accuracy. Plotting disabled.')
    for name in models:
        torch.manual_seed(1234)
        m = training.model(args.case, 'EXP')
        m.model_ini_dict['model'] = [name]
        if args.steps is not None:
            m.train_steps = args.steps
        if args.grid is not None:
            m.grid_node_num = args.grid
        # Periodic upstream saves rewrite complete loss histories. Keep final save only.
        m.pace_record_state = 0
        m.pace_record_gap = [max(m.train_steps, int(m.train_steps * m.train_ratio)) + 1]
        m.pace_record_skip = [0]
        m.save_desti = str(output / name) + '/'
        m.result_show = lambda: None
        if args.device == 'mps':
            torch.mps.synchronize()
        start = time.perf_counter()
        m.train()
        if args.device == 'mps':
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        r = dict(model=name, steps=m.train_steps, grid=m.grid_node_num,
                 interior_points=len(m.x), parameters=sum(x.numel() for x in m.net.parameters()),
                 elapsed_seconds=elapsed, training_seconds=m.time_list[0],
                 seconds_per_teacher_step=m.time_list[0] / m.net.iter,
                 first_loss=m.net.loss_list[0], final_loss=m.net.loss_list[-1],
                 peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024))
        if m.distill_state:
            r.update(student_steps=m.net_student.iter, student_seconds=m.time_list[1],
                     student_final_loss=m.net_student.loss_list[-1])
        if args.case == 'Laplace':
            with torch.no_grad():
                axis = torch.linspace(-1, 1, 101, device=training.device)
                x, y = torch.meshgrid(axis, axis, indexing='ij')
                xy = torch.stack([x.flatten(), y.flatten()], dim=1)
                u = m.net(xy).flatten()
                exact = xy[:, 0]**3 - 3 * xy[:, 0] * xy[:, 1]**2
                flipped = xy.clone(); flipped[:, 1] *= -1
                r['field_mse'] = ((u - exact)**2).mean().item()
                r['relative_l2'] = (torch.linalg.vector_norm(u-exact) / torch.linalg.vector_norm(exact)).item()
                r['even_y_max_error'] = (u - m.net(flipped).flatten()).abs().max().item()
        results.append(r)
        (output / 'summary.json').write_text(json.dumps(dict(metadata=metadata, results=results), indent=2))
        print(json.dumps(r), flush=True)
    print('Summary:', output / 'summary.json')


if __name__ == '__main__':
    main()
