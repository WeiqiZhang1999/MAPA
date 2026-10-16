"""One-dataset release example. Simulation is an integration check, not a benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def simulated_data(root):
    root.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split, count, seed in [('train', 24, 42), ('val', 8, 43), ('test', 8, 44)]:
        rng = np.random.default_rng(seed)
        grid = np.indices((28, 28, 28)).astype(np.float32)
        images = []
        labels = np.arange(count) % 2
        for label in labels:
            center = np.array([10 + label * 6, 14, 14]) + rng.normal(0, 1, 3)
            radius = rng.uniform(4, 6)
            sphere = ((grid - center[:, None, None, None]) ** 2).sum(0) < radius ** 2
            volume = np.clip(sphere * 180 + rng.normal(25, 8, sphere.shape), 0, 255)
            images.append(volume.astype(np.uint8))
        arrays[split + '_images'] = np.stack(images)
        arrays[split + '_labels'] = labels[:, None].astype(np.int64)
    np.savez_compressed(root / 'organmnist3d.npz', **arrays)
    (root / 'organmnist3d_summary.json').write_text(json.dumps({
        'task': 'binary-class', 'label': {'0': 'simulated-left', '1': 'simulated-right'},
        'simulated': True, 'not_a_benchmark': True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulate', action='store_true', help='Generate independent fake splits locally')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'organmnist3d_example')
    parser.add_argument('--include-baselines', action='store_true', help='Also run voxel DM/DC and progressive MTT')
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / 'completion.json').exists():
        raise FileExistsError('Use a fresh output directory to keep runs separate')
    data = (args.data_root or (output / 'simulated_data' if args.simulate else ROOT / 'data')).resolve()
    if args.simulate:
        if args.data_root:
            parser.error('--simulate owns its data directory. Do not pass --data-root.')
        simulated_data(data)
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'OPENBLAS_NUM_THREADS': '2', 'LOKY_MAX_CPU_COUNT': '2'}
    commands = []

    def run(script, options):
        command = [sys.executable, str(ROOT / 'scripts' / script)] + [str(x) for x in options]
        print('Running', script, flush=True)
        commands.append(command)
        with (output / (Path(script).stem + '_' + str(len(commands)) + '.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    shared = ['--dataset', 'organmnist3d', '--data-root', data, '--device', args.device,
              '--seed', 42, '--num-workers', 0]
    ae_args = ['--output-dir', output, '--epochs', 1 if args.simulate else 30,
               '--batch-size', 4, '--channels', '8,16' if args.simulate else '32,64',
               '--latent-channels', 16, '--block', 'plain' if args.simulate else 'residual',
               '--mse-weight', .05, '--grad-weight', 0 if args.simulate else .02,
               '--ssim-weight', 0 if args.simulate else .05]
    run('train_3d_autoencoder.py', shared + ae_args)
    checkpoint = output / 'autoencoder/organmnist3d_seed42/best.pt'
    evaluation = ['--eval-model', 'convnet3d', '--eval-width', 4 if args.simulate else 32,
                  '--eval-epochs', 1 if args.simulate else 100, '--eval-every', 1,
                  '--eval-batch-size', 4 if args.simulate else 32, '--eval-optimizer', 'sgd',
                  '--eval-lr', .01, '--eval-weight-decay', .0001, '--select-metric', 'balanced_acc']
    latent = shared + evaluation + ['--autoencoder-checkpoint', checkpoint, '--ipc', 1,
              '--lpc-multiplier', 4, '--latent-metric', 'dm', '--latent-init', 'random',
              '--latent-steps', 2 if args.simulate else 100, '--latent-lr', .005,
              '--dist-weight', .2, '--anchor-weight', 2, '--eval-seed', 101,
              '--dm-width', 4 if args.simulate else 64, '--dm-feature-dim', 8 if args.simulate else 128,
              '--dm-real-batch', 8 if args.simulate else 64, '--output-dir', output / 'mapa',
              '--method-name', 'MAPA-example', '--mode-cover-k', 2 if args.simulate else 4,
              '--mode-cover-allocation', 'uniform', '--mode-cover-weight', .10,
              '--mode-cover-strategy', 'softmax', '--mode-cover-temperature', .05,
              '--mode-cover-balance-weight', 1, '--mode-cover-anneal', 'cosine',
              '--mode-cover-anneal-start', 0, '--mode-cover-anneal-end', .7]
    run('distill_3d_latent_dd.py', latent)
    synthetic = next((output / 'mapa/runs').glob('*/synthetic_decoded.pt'))
    run('inspect_allocation.py', ['--data-root', data, '--checkpoint', checkpoint,
        '--synthetic', synthetic, '--anchors', output / 'anchors.pt', '--output', output / 'allocation.json',
        '--k', 2 if args.simulate else 4])
    if args.include_baselines:
        for method in ['dm', 'dc']:
            run('distill_3d_dm.py', shared + evaluation + ['--dd-method', method, '--ipc', 1,
                '--iterations', 1 if args.simulate else 100, '--batch-real', 4,
                '--dm-width', 4 if args.simulate else 32, '--output-dir', output / ('voxel_' + method)])
        run('distill_3d_mtt.py', shared + evaluation + ['--ipc', 1, '--num-experts', 1,
            '--expert-epochs', 1 if args.simulate else 3, '--expert-steps-per-epoch', 1 if args.simulate else 10,
            '--expert-batch-per-class', 2, '--mtt-iterations', 1 if args.simulate else 20,
            '--syn-steps', 1 if args.simulate else 3, '--expert-window', 1,
            '--mtt-width', 4 if args.simulate else 16, '--output-dir', output / 'voxel_mtt'])
    run('summarize_ae_storage_accounting.py', ['--repo-root', ROOT, '--ae-root', output / 'autoencoder',
        '--synthetic-root', output / 'mapa', '--output-dir', output / 'accounting'])
    completion = {'status': 'PASS', 'simulated': args.simulate, 'not_paper_results': True,
                  'device': args.device, 'commands': commands,
                  'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (output / 'completion.json').write_text(json.dumps(completion, indent=2))
    print(json.dumps({k: v for k, v in completion.items() if k != 'commands'}, indent=2))


if __name__ == '__main__':
    main()
