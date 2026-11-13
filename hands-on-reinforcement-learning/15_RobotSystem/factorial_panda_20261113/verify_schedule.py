"""Verify the scheduled experiment's frozen code, environment and output path; never train."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('expected_manifest_sha256')
    args = parser.parse_args()
    if digest(args.manifest) != args.expected_manifest_sha256:
        raise RuntimeError('Scheduled manifest changed; do not launch')
    data = json.loads(args.manifest.read_text(encoding='utf-8'))
    for name, expected in data['reviewed_files'].items():
        if digest(Path(name)) != expected:
            raise RuntimeError(f'Reviewed file changed: {name}')
    spec = importlib.util.spec_from_file_location('factorial_runner', HERE / 'runner.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.check_device('cuda')
    if runner.environment_snapshot(runner.SCENE) != data['environment']:
        raise RuntimeError('Runtime environment changed after scheduling')
    runner.protocol(device='cuda')
    if runner.ROOT.exists():
        raise FileExistsError(f'Result directory already exists: {runner.ROOT}')
    print('PASS: frozen files, CUDA environment, protocol and new result directory')


if __name__ == '__main__':
    main()
