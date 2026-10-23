"""显式下载 materials.json 中选定的资料；默认仅列出，不安装任何依赖。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(item):
    path = ROOT / item['path']
    if path.exists():
        if path.stat().st_size == item['bytes'] and digest(path) == item['sha256']:
            print('Verified:', item['path'])
            return
        raise RuntimeError(f'Existing file differs; inspect before replacing: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    try:
        with urlopen(item['url'], timeout=60) as response, partial.open('wb') as output:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                output.write(block)
        if partial.stat().st_size != item['bytes'] or digest(partial) != item['sha256']:
            raise RuntimeError('Downloaded bytes differ from recorded snapshot; keep original manifest and inspect source version')
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print('Downloaded and verified:', item['path'])


def checkout(item):
    path = ROOT / item['path']
    if (path / '.git').is_dir():
        head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if head != item['commit']:
            raise RuntimeError(f'Existing checkout has a different commit; preserve it and inspect: {path}')
        print('Commit verified:', item['path'])
        return
    if path.exists():
        raise RuntimeError(f'Existing directory is not the recorded checkout: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    # --no-checkout avoids executing examples or attempting to initialize a training environment.
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')  # 实际 SST 仅在显式 --sst 时下载。
    subprocess.run(['git', 'clone', '--depth', '1', '--no-checkout', item['url'], str(path)], check=True, env=env)
    subprocess.run(['git', '-C', str(path), 'fetch', '--depth', '1', 'origin', item['commit']], check=True, env=env)
    subprocess.run(['git', '-C', str(path), 'checkout', '--detach', item['commit']], check=True, env=env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codes', action='store_true', help='Fetch the three SHRED code repositories')
    p.add_argument('--pdebench-code', action='store_true')
    p.add_argument('--papers', action='store_true')
    p.add_argument('--sst', action='store_true', help='Fetch 187 MB SST example, not just its LFS pointer')
    p.add_argument('--sample', action='store_true', help='Fetch only the 4.95 MB PDEBench Sod6 example')
    args = p.parse_args()
    manifest = json.loads((ROOT / 'shred' / 'materials.json').read_text())
    if not any(vars(args).values()):
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    for item in manifest['repositories']:
        if (args.codes and item['name'] != 'pdebench') or (args.pdebench_code and item['name'] == 'pdebench'):
            checkout(item)
    for item in manifest['files']:
        if (args.papers and item['path'].endswith('.pdf')) or (args.sst and item['path'].endswith('.mat')) or (args.sample and item['path'].endswith('.hdf5')):
            download(item)


if __name__ == '__main__':
    main()
