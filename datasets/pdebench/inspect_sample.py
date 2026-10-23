"""核对小样本的官方校验值；已有 h5py 时额外列出字段和形状，不自动安装。"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    path = root / 'data' / 'Sod6.hdf5'
    if not path.is_file():
        raise SystemExit('Missing Sod6.hdf5; see shred/materials.json and fetch_materials.py --sample')
    with (root / 'catalog.csv').open() as f:
        item = next(row for row in csv.DictReader(f) if row['file_id'] == '133150')
    h = hashlib.md5()
    with path.open('rb') as f:
        magic = f.read(8)
        f.seek(0)
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    assert path.stat().st_size == int(item['bytes']), 'Size mismatch'
    assert h.hexdigest() == item['checksum'], 'Checksum mismatch'
    assert magic == b'\x89HDF\r\n\x1a\n', 'Invalid HDF5 signature'
    report = dict(file='Sod6.hdf5', bytes=path.stat().st_size, md5=h.hexdigest(),
                  checksum_verified=True, hdf5_signature_verified=True)
    if importlib.util.find_spec('h5py') is None:
        report['inspection'] = 'header/checksum only; h5py is not installed; array shapes are NOT inspected'
    else:
        import h5py
        fields = {}
        with h5py.File(path, 'r') as f:
            def visit(name, item):
                if isinstance(item, h5py.Dataset):
                    fields[name] = dict(shape=list(item.shape), dtype=str(item.dtype))
            f.visititems(visit)
        report.update(inspection='dataset names/shapes/dtypes; no full arrays loaded', fields=fields)
    output = root / 'Results' / 'sample-inspection.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
