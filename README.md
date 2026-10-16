# MAPA

Code release package for **MAPA: Mode-Aware Prototype Allocation for Latent Dataset Distillation in 3D Medical Classification**, WACV 2027.

Weiqi Zhang and Hiroyuki Kubo, Chiba University.

This package covers one dataset example, OrganMNIST3D. It contains the research implementations and a small CPU simulation that exercises the interfaces. The simulated volumes, checkpoint and validation outputs are **not** OrganMNIST3D observations, paper results, or a pretrained medical model.

Repository: [WeiqiZhang1999/MAPA](https://github.com/WeiqiZhang1999/MAPA).

## Release contents

| Component | Files | Scope |
|---|---|---|
| MAPA and latent matching | `scripts/distill_3d_latent_dd.py` | Fixed train-latent anchors, soft allocation, anti-collapse and curriculum settings |
| Autoencoder | `scripts/train_3d_autoencoder.py` | Train on training data, select reconstruction checkpoint on validation data |
| Example AE weights | `checkpoints/simulated_organ_schema_ae.pt` | Trained only on fake binary volumes, for interface validation |
| 3D DC and DM adaptations | `scripts/distill_3d_dm.py --dd-method dc` or `dm` | Voxel-space gradient/distribution matching |
| 3D MTT adaptation | `scripts/distill_3d_mtt.py` | Progressive short-trajectory matching, not an exact reproduction of all original MTT configurations |
| Latent MTT | `scripts/distill_3d_latent_mtt.py` | Research implementation, not exercised by the short integration example |
| Evaluators | `src/models3d.py`, `scripts/train_medmnist3d_cls.py` | ConvNet3D and alternative 3D backbones |
| Shared K-means anchors and PCI | `scripts/inspect_allocation.py` | Cluster training latents only, cache anchors, compute hard occupancy and entropy diagnostic |
| Byte accounting | `scripts/summarize_ae_storage_accounting.py` | Separate per-prototype tensors, decoder weights, AE training and downstream costs |

Raw medical data, patient identifiers, original training examples, and the full five-dataset checkpoint collection are not distributed. The demo checkpoint contains weights and architecture metadata, with no reconstruction examples or local filesystem paths. Paper AE weights are not included in this initial single-example package.

## Environment

Use Python 3.10 or newer. Install the appropriate official PyTorch build for your platform before installing the remaining packages.

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
```

The local integration check used Python 3.11.15, torch 2.6.0+cu124, scikit-learn 1.8.0 and CPU execution. Exact package versions are recorded in `validation/simulation.json` after validation. CUDA execution and formal benchmark reproduction are not validated by this smoke check.

## CPU simulation

Run from the repository root. This creates 24 training, 8 validation and 8 test fake volumes with independent random generators. The binary labels describe sphere location. The NPZ has the OrganMNIST3D loader schema, but is not the real 11-class dataset.

```bash
python examples/run_organmnist3d.py --simulate --device cpu --include-baselines
```

The example trains a tiny AE for one epoch, optimizes MAPA for two steps, trains a fresh evaluator for one epoch, computes shared-anchor occupancy, runs one-step voxel DM/DC/MTT checks, and writes storage accounting. Expect a short integration run, not useful accuracy.

Outputs are under `runs/organmnist3d_example/`.

- `completion.json` is written with status PASS only if every requested subprocess succeeds.
- `autoencoder/organmnist3d_seed42/best.pt` is the simulated AE.
- `mapa/runs/*/synthetic_decoded.pt` contains latent prototypes and decoded fake volumes.
- `anchors.pt` and `allocation.json` contain fixed train-latent clusters and occupancy diagnostics. The local cache contains training latents and is excluded from publication.
- `accounting/ae_cost_accounting.csv` and `accounting/storage_accounting.csv` contain separate accounting rows.
- Each stage writes its own log. Use a fresh `--output` directory for a new run.

Export the example weights without training samples or machine paths after successful simulation.

```bash
python scripts/export_demo_checkpoint.py --run runs/organmnist3d_example --output checkpoints/simulated_organ_schema_ae.pt
```

## Real OrganMNIST3D example

For a single real-data example, run without `--simulate`. Supply an unmodified local `organmnist3d.npz`, or allow the MedMNIST loader to download it when the data directory is empty. Obtain and use the data under the original dataset terms. Keep official train, validation and test splits separate.

```bash
python examples/run_organmnist3d.py --device cuda --data-root data --output runs/organmnist3d_real_example --include-baselines
```

This uses the paper-style OrganMNIST3D AE geometry with channels 32/64, 16 latent channels, a 7 cubed spatial latent, residual refinement, and K=4. It runs IPC=1 with four latent prototypes per voxel prototype. The full benchmark also requires five seeds 42 to 46, IPC 1 and 10 and method-specific baseline schedules. The one-seed example is not a command to regenerate every paper table. See the paper and supplementary material for the full protocol. Real-data execution has not been run as part of release validation.

The original research CLI arguments are retained for traceability. For example, `mode_cover_*` refers to MAPA allocation, `dist_weight` scales the class-matching objective, and `anchor_weight` is the research implementation's latent-initialization penalty. The example does not silently remove these settings.

## Comparing methods

Use the same AE, split, seeds, prototype count, evaluator, learning rate, weight decay and validation-checkpoint rule. With 28 cubed float32 volumes and 16 by 7 cubed float32 latents, the per-prototype byte ratio is 4. Decoder weights are a separate one-time cost, not silently folded into that ratio. Decoded-volume count and total end-to-end time must also be reported separately.

For class-level Latent-DM, use `distill_3d_latent_dd.py` with `--mode-cover-k 0 --mode-cover-weight 0` and otherwise matched settings. The same script exposes latent gradient matching and MMD via `--latent-metric gm` and `mmd`. MAPA uses `--latent-metric dm --mode-cover-k 4 --mode-cover-strategy softmax --mode-cover-temperature 0.05 --mode-cover-balance-weight 1 --mode-cover-weight 0.10 --mode-cover-anneal cosine --mode-cover-anneal-end 0.7` for this real-data example.

Reuse exactly the same `--anchors` cache for all methods in an occupancy comparison. The diagnostic exporter rejects a cache if the encoder's training latents, labels, K or seed change. PCI is a distribution diagnostic, not an accuracy predictor. It is undefined for K=1. No test features or test labels are used to fit anchors or optimize prototypes.

The paper's severity analysis is a separate readout and is not reproduced by this OrganMNIST3D example.

## Provenance and references

`SOURCE_MANIFEST.json` records the source hashes of the copied research implementations. Added example and export scripts orchestrate those implementations without changing their objectives. The release does not include the manuscript, internal experiment archives, private paths, or unpublished result tables.

Please cite the MAPA paper using the publisher's final bibliographic record when available. The baseline methods derive from Dataset Condensation with Gradient Matching (Zhao et al.), Dataset Condensation with Distribution Matching (Zhao and Bilen), Dataset Distillation by Matching Training Trajectories (Cazenavette et al.), and Dataset Distillation in Latent Space (Duan et al.). Data provenance is MedMNIST v2 (Yang et al.). These adaptations are implemented in 3D and should be reported as adaptations rather than official upstream releases.

License selection is pending author confirmation. No additional license grant is implied by this preparation package.
