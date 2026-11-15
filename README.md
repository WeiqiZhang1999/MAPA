<h1 align="center">MAPA</h1>
<h3 align="center">Mode-Aware Prototype Allocation for Latent Dataset Distillation<br>in 3D Medical Classification</h3>

<p align="center">
  <b>WACV 2027</b><br>
  Weiqi Zhang &nbsp;&nbsp; Hiroyuki Kubo<br>
  Chiba University
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.09956">Paper</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/pdf/2611.09956">PDF</a> &nbsp;|&nbsp;
  <a href="#quick-demo">Quick demo</a> &nbsp;|&nbsp;
  <a href="docs/demo.md">Demo guide</a>
</p>

<p align="center"><img src="assets/teaser.svg" width="900" alt="Class-level matching can concentrate prototypes; MAPA allocates them across intra-class modes."></p>

MAPA allocates a compact set of latent prototypes across intra-class modes for 3D medical classification. Fixed anchors estimate these modes from real training latents, while class-level matching is preserved.

## Method

<p align="center"><img src="assets/pipeline.svg" width="1000" alt="MAPA pipeline with a frozen autoencoder, fixed anchors, soft allocation and anti-collapse regularization."></p>

Volumes are encoded into a frozen latent space. MAPA optimizes prototypes through class-level matching and soft allocation, then decodes them for downstream training.

## Prototype allocation

<p align="center"><img src="assets/allocation.png" width="900" alt="Per-anchor prototype allocation, concentration diagnostic and effective modes on MosMedData at IPC10."></p>

*MosMedData, IPC 10. MAPA spreads the prototype budget more evenly across the fixed anchors. PCI is a concentration diagnostic.*

## Quick demo

```bash
git clone https://github.com/WeiqiZhang1999/MAPA.git
cd MAPA
python -m pip install -r requirements.txt
python demo/examples/run_organmnist3d.py --simulate --device cpu
```

The CPU demo uses simulated volumes with the OrganMNIST3D data schema. It checks the pipeline and does not reproduce the paper's benchmark results. See the [demo guide](docs/demo.md) for optional 3D baselines and output files.

## Paper

[**MAPA: Mode-Aware Prototype Allocation for Latent Dataset Distillation in 3D Medical Classification**](https://arxiv.org/abs/2611.09956)<br>
Weiqi Zhang and Hiroyuki Kubo<br>
Accepted at **WACV 2027**.

[arXiv](https://arxiv.org/abs/2611.09956) · [PDF](https://arxiv.org/pdf/2611.09956)

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{zhang2026mapa,
  title         = {{MAPA}: Mode-Aware Prototype Allocation for Latent Dataset Distillation in {3D} Medical Classification},
  author        = {Weiqi Zhang and Hiroyuki Kubo},
  year          = {2026},
  eprint        = {2611.09956},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  note          = {Accepted at WACV 2027},
  url           = {https://arxiv.org/abs/2611.09956}
}
```
