"""Compute fixed train-latent K-means anchors and hard-occupancy diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from distill_3d_latent_dd import load_autoencoder
from train_medmnist3d_cls import load_medmnist_dataset, prepare_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='organmnist3d')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--synthetic', type=Path, required=True)
    parser.add_argument('--anchors', type=Path, required=True,
                        help='Shared anchor cache. Reuse this path for all compared methods.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.k < 2:
        parser.error('PCI is undefined for K < 2')
    torch.set_num_threads(2)
    payload = torch.load(args.synthetic, map_location='cpu', weights_only=False)
    z = payload.get('z_final', payload.get('z_syn'))
    if z is None:
        raise ValueError('Synthetic checkpoint must contain z_final or z_syn')
    labels = payload['y_syn'].numpy()
    x, y = prepare_tensors(load_medmnist_dataset(args.dataset, 'train', args.data_root))
    ae, _ = load_autoencoder(args.checkpoint, torch.device('cpu'))
    with torch.no_grad():
        real = torch.cat([ae.encode(x[i:i+8]) for i in range(0, len(x), 8)]).flatten(1).numpy()
    if args.anchors.exists():
        cache = torch.load(args.anchors, map_location='cpu', weights_only=False)
        if cache['dataset'] != args.dataset or cache['k'] != args.k or cache['seed'] != args.seed:
            raise ValueError('Anchor cache configuration mismatch')
        if not torch.equal(cache['real_latents'], torch.from_numpy(real)) or not torch.equal(cache['labels'], y):
            raise ValueError('Training data or encoder changed. Use a fresh anchor cache.')
        anchors = cache['anchors']
    else:
        anchors = {}
        for c in np.unique(y.numpy()):
            group = real[y.numpy() == c]
            if len(group) < args.k:
                raise ValueError('Not enough training samples for K')
            anchors[int(c)] = KMeans(args.k, random_state=args.seed, n_init=10).fit(group).cluster_centers_
        args.anchors.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'dataset': args.dataset, 'k': args.k, 'seed': args.seed,
                    'real_latents': torch.from_numpy(real), 'labels': y, 'anchors': anchors}, args.anchors)
    rows = []
    for c in np.unique(labels):
        group = z[labels == c].flatten(1).numpy()
        distances = ((group[:, None, :] - anchors[int(c)][None, :, :]) ** 2).sum(-1)
        counts = np.bincount(distances.argmin(1), minlength=args.k)
        probabilities = counts / counts.sum()
        positive = probabilities[probabilities > 0]
        entropy = float(-(positive * np.log(positive)).sum())
        rows.append({'class': int(c), 'counts': counts.tolist(),
                     'pci': 1.0 - entropy / np.log(args.k), 'eff_modes': np.exp(entropy)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'diagnostic_only': True, 'classes': rows}, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
