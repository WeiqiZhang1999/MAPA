# Demo guide

This repository provides selected demonstration code and figures from MAPA. The example uses independent simulated binary volumes in the OrganMNIST3D loader schema, rather than the real 11-class medical dataset.

## Run

Use Python 3.10 or newer and install `requirements.txt` from the repository root.

```bash
python demo/examples/run_organmnist3d.py --simulate --device cpu
```

The example trains a small autoencoder, optimizes latent prototypes, trains an evaluator, exports fixed training-latent K-means anchors and PCI diagnostics, and reports tensor/decoder storage separately.

To also exercise the selected voxel-space DM, DC and progressive MTT adaptations:

```bash
python demo/examples/run_organmnist3d.py --simulate --device cpu --include-baselines --output runs/demo_with_baselines
```

Use a fresh output folder for each completed run. Without `--output`, files are saved under `demo/runs/organmnist3d_example/`.

## Files

```text
assets/                   Figures used in the README
demo/
  examples/               One-dataset demonstration entry point
  scripts/                Selected training, allocation and accounting routines
  src/                    3D network definitions
  checkpoints/            Simulated example AE weights
docs/
  demo.md                 This guide
  simulation.json         Integration-check record
  source_manifest.json    Source-file provenance
```

The example checkpoint is `demo/checkpoints/simulated_organ_schema_ae.pt`. It contains weights and architecture metadata trained only on simulated volumes. Paper-trained AE checkpoints and the complete experiment suite are not part of this demo.

Each successful run writes `completion.json`, stage logs, decoded prototypes, `allocation.json`, and an `accounting/` directory. The anchor cache is fitted only on training latents and should be shared when comparing allocation diagnostics. PCI is not an accuracy predictor.

The included research routines retain their original settings. This compact package is an integration demonstration, not a claim that every implementation setting reproduces the manuscript. No medical training data or internal experiment archives are included.

## References

The baseline adaptations build on gradient matching (Zhao et al.), distribution matching (Zhao and Bilen), trajectory matching (Cazenavette et al.) and latent distillation (Duan et al.). The simulated schema follows the MedMNIST loader format. The MTT adaptation uses progressive short trajectories.

README presentation takes inspiration from [MTT, CVPR 2022](https://github.com/GeorgeCazenavette/mtt-distillation) and [Dataset Condensation, ICLR 2021 / ICML 2021](https://github.com/VICO-UoE/DatasetCondensation). All displayed figures are MAPA's own local paper assets.

License selection remains pending author confirmation.
