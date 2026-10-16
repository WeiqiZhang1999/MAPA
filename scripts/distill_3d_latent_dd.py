from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from distill_3d_dm import append_dm_csv, calibrate_batchnorm_stats, eval_synthetic, freeze_batchnorm_stats, match_gradients  # noqa: E402
from models3d import build_model  # noqa: E402
from train_3d_autoencoder import ConvAutoencoder3D  # noqa: E402
from train_3d_autoencoder import volume_slice_uint8  # noqa: E402
from train_medmnist3d_cls import (  # noqa: E402
    get_dataset_info,
    infer_task,
    load_medmnist_dataset,
    make_loader,
    prepare_tensors,
    seed_everything,
)
from PIL import Image, ImageDraw


class LatentEmbedder3D(nn.Module):
    """Small random network used by LatentDM-style feature distribution matching."""

    def __init__(self, in_channels: int, width: int = 64, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, width)), width),
            nn.SiLU(inplace=True),
            nn.Conv3d(width, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, width)), width),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Linear(width, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).flatten(1))


def class_indices(y: torch.Tensor, num_classes: int) -> list[torch.Tensor]:
    return [torch.nonzero(y == c, as_tuple=False).flatten() for c in range(num_classes)]


def remap_legacy_plain_ae_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map early plain-AE checkpoint names to the current module layout."""
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix, mid in (("encoder", "down"), ("decoder", "up")):
            for idx in range(8):
                old_base = f"{prefix}.{idx}.block"
                if new_key.startswith(f"{old_base}.0."):
                    new_key = new_key.replace(f"{old_base}.0.", f"{prefix}.{idx}.{mid}.0.", 1)
                    break
                if new_key.startswith(f"{old_base}.1."):
                    new_key = new_key.replace(f"{old_base}.1.", f"{prefix}.{idx}.{mid}.1.", 1)
                    break
                if new_key.startswith(f"{old_base}.3."):
                    new_key = new_key.replace(f"{old_base}.3.", f"{prefix}.{idx}.refine.block.0.", 1)
                    break
                if new_key.startswith(f"{old_base}.4."):
                    new_key = new_key.replace(f"{old_base}.4.", f"{prefix}.{idx}.refine.block.1.", 1)
                    break
        remapped[new_key] = value
    return remapped


def load_autoencoder(checkpoint: Path, device: torch.device) -> tuple[ConvAutoencoder3D, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["model_state"]
    block = payload.get("block", "plain")
    if any(key.startswith("encoder.0.block.") for key in state_dict):
        block = "plain"
        state_dict = remap_legacy_plain_ae_state(state_dict)
    model = ConvAutoencoder3D(
        in_channels=payload["input_shape"][0],
        channels=tuple(payload["channels"]),
        latent_channels=int(payload["latent_channels"]),
        block=block,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, payload


def latent_storage_ratio(ae_payload: dict) -> float:
    input_shape = ae_payload.get("input_shape")
    latent_shape = ae_payload.get("latent_shape")
    if input_shape is None or latent_shape is None:
        return float("nan")
    input_values = [int(v) for v in input_shape]
    latent_values = [int(v) for v in latent_shape]
    input_storage = float(np.prod(input_values))
    latent_storage = float(np.prod(latent_values))
    return input_storage / max(latent_storage, 1.0)


@torch.no_grad()
def encode_volume_bank(model: ConvAutoencoder3D, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    latents = []
    for start in tqdm(range(0, x.shape[0], batch_size), desc="encode", leave=False):
        xb = x[start : start + batch_size].to(device, non_blocking=True)
        latents.append(model.encode(xb).detach().cpu())
    return torch.cat(latents, dim=0)


def init_latents(
    z_real: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    ipc: int,
    lpc_multiplier: int,
    seed: int,
    init: str,
    mixed_random_frac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    effective_ipc = ipc * lpc_multiplier
    z_syn = []
    y_syn = []
    for c, idx in enumerate(class_indices(y_train, num_classes)):
        if idx.numel() < effective_ipc and init != "random":
            raise ValueError(f"Class {c} has only {idx.numel()} samples, cannot init decoded_ipc={effective_ipc}")
        zc = z_real[idx]
        if init == "random":
            if idx.numel() >= effective_ipc:
                perm = torch.randperm(idx.numel(), generator=generator)[:effective_ipc]
            else:
                perm = torch.randint(idx.numel(), (effective_ipc,), generator=generator)
            centers = zc[perm].clone()
        elif init == "kmeans":
            flat = zc.flatten(1).numpy()
            kmeans = KMeans(n_clusters=effective_ipc, n_init=10, random_state=seed)
            centers = torch.from_numpy(kmeans.fit(flat).cluster_centers_).float().reshape(effective_ipc, *zc.shape[1:])
        elif init == "mixed":
            n_random = int(round(effective_ipc * mixed_random_frac))
            n_random = max(1, min(effective_ipc - 1, n_random))
            n_kmeans = effective_ipc - n_random
            flat = zc.flatten(1).numpy()
            kmeans = KMeans(n_clusters=n_kmeans, n_init=10, random_state=seed)
            kmeans_centers = torch.from_numpy(kmeans.fit(flat).cluster_centers_).float().reshape(n_kmeans, *zc.shape[1:])
            perm = torch.randperm(idx.numel(), generator=generator)[:n_random]
            random_centers = zc[perm].clone()
            centers = torch.cat([kmeans_centers, random_centers], dim=0)
        else:
            raise ValueError(f"Unknown latent init: {init}")
        z_syn.append(centers)
        y_syn.append(torch.full((effective_ipc,), c, dtype=torch.long))
    return torch.cat(z_syn, dim=0), torch.cat(y_syn, dim=0)


def init_mode_covered_latents(
    z_real: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    ipc: int,
    lpc_multiplier: int,
    seed: int,
    num_modes: int,
    allocation_mode: str,
    anchor_init: str,
    prototype_init: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor | None]]:
    effective_ipc = ipc * lpc_multiplier
    if num_modes <= 0:
        raise ValueError("--mode-cover-k must be positive for mode-covered initialization")
    if num_modes > effective_ipc:
        raise ValueError(f"--mode-cover-k={num_modes} cannot exceed decoded_ipc={effective_ipc}")
    generator = torch.Generator().manual_seed(seed)
    z_syn = []
    y_syn = []
    mode_syn = []
    real_mode_assignments = []
    mode_center_overrides = []
    for c, idx in enumerate(class_indices(y_train, num_classes)):
        if idx.numel() < num_modes:
            raise ValueError(f"Class {c} has only {idx.numel()} samples, cannot form {num_modes} modes")
        zc = z_real[idx]
        flat = zc.flatten(1).numpy()
        if anchor_init == "kmeans":
            kmeans = KMeans(n_clusters=num_modes, n_init=10, random_state=seed)
            assignments_np = kmeans.fit_predict(flat)
            mode_center_overrides.append(None)
        elif anchor_init == "random":
            anchor_idx = torch.randperm(idx.numel(), generator=generator)[:num_modes].numpy()
            centers_np = flat[anchor_idx]
            diff = flat[:, None, :] - centers_np[None, :, :]
            assignments_np = np.argmin(np.sum(diff * diff, axis=2), axis=1)
            mode_center_overrides.append(torch.from_numpy(centers_np).float())
        else:
            raise ValueError(f"Unknown mode anchor init: {anchor_init}")
        assignments = torch.from_numpy(assignments_np).long()
        real_mode_assignments.append(assignments)
        counts = torch.bincount(assignments, minlength=num_modes).float()
        if allocation_mode == "uniform":
            base = effective_ipc // num_modes
            remainder = effective_ipc % num_modes
            allocation = torch.full((num_modes,), base, dtype=torch.long)
            if remainder > 0:
                allocation[:remainder] += 1
        elif allocation_mode == "proportional":
            allocation = torch.floor(counts / counts.sum().clamp_min(1.0) * effective_ipc).long()
            allocation = torch.maximum(allocation, torch.ones_like(allocation))
        else:
            raise ValueError(f"Unknown mode allocation: {allocation_mode}")
        while int(allocation.sum().item()) > effective_ipc:
            candidates = torch.nonzero(allocation > 1, as_tuple=False).flatten()
            if candidates.numel() == 0:
                break
            chosen = candidates[torch.argmax(allocation[candidates])]
            allocation[chosen] -= 1
        while int(allocation.sum().item()) < effective_ipc:
            residual = counts / counts.sum().clamp_min(1.0) * effective_ipc - allocation.float()
            chosen = torch.argmax(residual)
            allocation[chosen] += 1
        for mode_id in range(num_modes):
            local = torch.nonzero(assignments == mode_id, as_tuple=False).flatten()
            take = int(allocation[mode_id].item())
            if prototype_init == "random":
                if local.numel() >= take:
                    perm = local[torch.randperm(local.numel(), generator=generator)[:take]]
                else:
                    perm = local[torch.randint(local.numel(), (take,), generator=generator)]
            elif prototype_init == "medoid":
                if local.numel() == 0:
                    raise ValueError(f"Class {c} mode {mode_id} has no assigned real latents")
                if anchor_init == "kmeans":
                    center = torch.from_numpy(kmeans.cluster_centers_[mode_id]).float()
                else:
                    center = torch.from_numpy(centers_np[mode_id]).float()
                local_flat = zc[local].flatten(1)
                dist = torch.sum((local_flat - center.unsqueeze(0)) ** 2, dim=1)
                order = local[torch.argsort(dist)]
                if order.numel() >= take:
                    perm = order[:take]
                else:
                    repeats = int(np.ceil(take / order.numel()))
                    perm = order.repeat(repeats)[:take]
            else:
                raise ValueError(f"Unknown mode prototype init: {prototype_init}")
            z_syn.append(zc[perm].clone())
            y_syn.append(torch.full((take,), c, dtype=torch.long))
            mode_syn.append(torch.full((take,), mode_id, dtype=torch.long))
    return (
        torch.cat(z_syn, dim=0),
        torch.cat(y_syn, dim=0),
        torch.cat(mode_syn, dim=0),
        real_mode_assignments,
        mode_center_overrides,
    )


def shuffle_mode_assignments(
    mode_assignments: list[torch.Tensor],
    num_modes: int,
    seed: int,
) -> list[torch.Tensor]:
    """Deterministically break prototype-to-mode correspondence for ablations."""

    shuffled = []
    base = torch.arange(num_modes)
    for class_id, assignments in enumerate(mode_assignments):
        generator = torch.Generator().manual_seed(seed + 7919 * (class_id + 1))
        perm = torch.randperm(num_modes, generator=generator)
        if torch.equal(perm, base):
            perm = torch.roll(perm, shifts=1)
        shuffled.append(perm[assignments.long()])
    return shuffled


def pairwise_sqdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    return (x2 + y2 - 2.0 * x @ y.T).clamp_min(0.0)


def estimate_sigma2(x: torch.Tensor, max_points: int = 256) -> float:
    if x.shape[0] > max_points:
        perm = torch.randperm(x.shape[0], device=x.device)[:max_points]
        x = x[perm]
    d2 = pairwise_sqdist(x, x)
    values = d2[d2 > 0]
    if values.numel() == 0:
        return 1.0
    return float(values.median().clamp_min(1e-6).item())


def rbf_mmd_loss(syn: torch.Tensor, real: torch.Tensor, real_kernel_mean: torch.Tensor, sigma2: float) -> torch.Tensor:
    k_ss = torch.exp(-pairwise_sqdist(syn, syn) / (2.0 * sigma2)).mean()
    k_sr = torch.exp(-pairwise_sqdist(syn, real) / (2.0 * sigma2)).mean()
    return k_ss + real_kernel_mean - 2.0 * k_sr


def sinkhorn_ot_loss(syn: torch.Tensor, real: torch.Tensor, epsilon: float, iters: int) -> torch.Tensor:
    cost = pairwise_sqdist(syn, real) / max(float(syn.shape[1]), 1.0)
    return sinkhorn_from_cost(cost, epsilon, iters)


def sinkhorn_from_cost(cost: torch.Tensor, epsilon: float, iters: int) -> torch.Tensor:
    n = cost.shape[0]
    m = cost.shape[1]
    log_a = torch.full((n,), -math.log(n), device=cost.device, dtype=cost.dtype)
    log_b = torch.full((m,), -math.log(m), device=cost.device, dtype=cost.dtype)
    log_k = -cost / epsilon
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(iters):
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_k + u.unsqueeze(1), dim=0)
    plan = torch.exp(log_k + u.unsqueeze(1) + v.unsqueeze(0))
    return (plan * cost).sum()


def cosine_sinkhorn_ot_loss(syn: torch.Tensor, real: torch.Tensor, epsilon: float, iters: int) -> torch.Tensor:
    syn_n = F.normalize(syn, dim=1)
    real_n = F.normalize(real, dim=1)
    cost = (1.0 - syn_n @ real_n.T).clamp_min(0.0)
    return sinkhorn_from_cost(cost, epsilon, iters)


def prepare_latent_stats(
    z_real: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    device: torch.device,
    metric: str,
    local_token_weight_mode: str,
    local_token_weight_power: float,
    difficulty_mode: str,
    difficulty_hard_quantile: float,
    mode_cover_k: int,
    mode_cover_allocation: str,
    mode_assignments: list[torch.Tensor] | None = None,
    mode_center_overrides: list[torch.Tensor | None] | None = None,
) -> list[dict]:
    stats = []
    flat_all = z_real.to(device).flatten(1)
    y_device = y_train.to(device)
    centroids = []
    for c in range(num_classes):
        centroids.append(flat_all[y_device == c].mean(dim=0))
    centroids_tensor = torch.stack(centroids, dim=0)
    centroids_n = F.normalize(centroids_tensor, dim=1)
    for c, idx in enumerate(class_indices(y_train, num_classes)):
        zc_tensor = z_real[idx].to(device)
        zc = zc_tensor.flatten(1)
        item = {"real": zc, "real_tensor": zc_tensor}
        if mode_cover_k > 0:
            if mode_assignments is not None:
                assignments = mode_assignments[c].long().to(device)
            else:
                flat_cpu = z_real[idx].flatten(1).numpy()
                kmeans = KMeans(n_clusters=mode_cover_k, n_init=10, random_state=0)
                assignments = torch.from_numpy(kmeans.fit_predict(flat_cpu)).long().to(device)
            item["mode_assignments"] = assignments.detach()
            item["num_modes"] = mode_cover_k
            counts = torch.bincount(assignments, minlength=mode_cover_k).float().to(device)
            if mode_cover_allocation == "uniform":
                mode_prior = torch.full((mode_cover_k,), 1.0 / mode_cover_k, device=device)
            elif mode_cover_allocation == "proportional":
                mode_prior = counts / counts.sum().clamp_min(1.0)
            else:
                raise ValueError(f"Unknown mode allocation: {mode_cover_allocation}")
            if mode_center_overrides is not None and mode_center_overrides[c] is not None:
                mode_centers = mode_center_overrides[c].to(device)
            else:
                mode_centers = []
                for mode_id in range(mode_cover_k):
                    mode_latents = zc[assignments == mode_id]
                    if mode_latents.numel() == 0:
                        mode_centers.append(zc.mean(dim=0))
                    else:
                        mode_centers.append(mode_latents.mean(dim=0))
                mode_centers = torch.stack(mode_centers, dim=0)
            item["mode_prior"] = mode_prior.detach()
            item["mode_centers"] = mode_centers.detach()
        else:
            item["mode_assignments"] = torch.zeros(zc.shape[0], dtype=torch.long, device=device)
            item["num_modes"] = 0
            item["mode_prior"] = torch.empty(0, device=device)
            item["mode_centers"] = torch.empty(0, zc.shape[1], device=device)
        if difficulty_mode != "none":
            zc_n = F.normalize(zc, dim=1)
            sims = zc_n @ centroids_n.T
            own_sim = sims[:, c]
            if num_classes > 1:
                other_mask = torch.ones(num_classes, dtype=torch.bool, device=device)
                other_mask[c] = False
                max_other_sim = sims[:, other_mask].max(dim=1).values
            else:
                max_other_sim = torch.zeros_like(own_sim)
            if difficulty_mode == "centroid_margin":
                difficulty = max_other_sim - own_sim
            elif difficulty_mode == "centroid_distance":
                difficulty = 1.0 - own_sim
            else:
                raise ValueError(f"Unknown difficulty mode: {difficulty_mode}")
            hard_quantile = min(max(float(difficulty_hard_quantile), 0.0), 1.0)
            if hard_quantile <= 0.0:
                hard_idx = torch.empty(0, dtype=torch.long, device=device)
            elif hard_quantile >= 1.0:
                hard_idx = torch.arange(zc.shape[0], device=device)
            else:
                threshold = torch.quantile(difficulty, 1.0 - hard_quantile)
                hard_idx = torch.nonzero(difficulty >= threshold, as_tuple=False).flatten()
            item["difficulty"] = difficulty.detach()
            item["hard_idx"] = hard_idx.detach()
        else:
            item["difficulty"] = torch.zeros(zc.shape[0], device=device)
            item["hard_idx"] = torch.empty(0, dtype=torch.long, device=device)
        local_tokens = zc_tensor.flatten(2).transpose(1, 2).reshape(-1, zc_tensor.shape[1])
        item["real_token_mean"] = local_tokens.mean(dim=0).detach()
        item["real_token_std"] = local_tokens.std(dim=0, unbiased=False).detach()
        weighted_mean, weighted_std = weighted_token_stats(
            local_tokens,
            mode=local_token_weight_mode,
            power=local_token_weight_power,
        )
        item["real_token_weighted_mean"] = weighted_mean.detach()
        item["real_token_weighted_std"] = weighted_std.detach()
        if metric == "mmd":
            sigma2 = estimate_sigma2(zc)
            item["sigma2"] = sigma2
            with torch.no_grad():
                item["real_kernel_mean"] = torch.exp(-pairwise_sqdist(zc, zc) / (2.0 * sigma2)).mean()
        stats.append(item)
    return stats


def save_synthetic_grid(x_syn: torch.Tensor, y_syn: torch.Tensor, path: Path, max_items: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_syn = x_syn.detach().cpu()
    y_syn = y_syn.detach().cpu()
    classes = sorted(int(c) for c in torch.unique(y_syn).tolist())
    selected = []
    per_class = max(1, max_items // max(len(classes), 1))
    for c in classes:
        idx = torch.nonzero(y_syn == c, as_tuple=False).flatten()
        selected.extend(idx[:per_class].tolist())
    selected = selected[:max_items]
    if not selected:
        selected = list(range(min(max_items, x_syn.shape[0])))
    x_syn = x_syn[selected]
    y_syn = y_syn[selected]
    n = x_syn.shape[0]
    tile_w = int(x_syn.shape[-1])
    tile_h = int(x_syn.shape[-2])
    scale = max(1, min(4, 112 // max(tile_w, 1)))
    disp_w = tile_w * scale
    disp_h = tile_h * scale
    label_h = 18
    cols = min(8, n)
    rows = int(math.ceil(n / cols))
    canvas = Image.new("RGB", (cols * disp_w, rows * (disp_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx in range(n):
        row = idx // cols
        col = idx % cols
        x0 = col * disp_w
        y0 = row * (disp_h + label_h)
        draw.text((x0 + 2, y0 + 2), f"y={int(y_syn[idx])}", fill=(0, 0, 0))
        canvas.paste(volume_slice_uint8(x_syn[idx]).resize((disp_w, disp_h), Image.Resampling.NEAREST), (x0, y0 + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def latent_distribution_loss(
    z_syn: torch.Tensor,
    y_syn: torch.Tensor,
    stats: list[dict],
    num_classes: int,
    metric: str,
    ot_epsilon: float,
    ot_iters: int,
) -> torch.Tensor:
    losses = []
    flat_syn = z_syn.flatten(1)
    for c in range(num_classes):
        syn = flat_syn[y_syn == c]
        real = stats[c]["real"]
        if metric == "mmd":
            losses.append(rbf_mmd_loss(syn, real, stats[c]["real_kernel_mean"], stats[c]["sigma2"]))
        elif metric == "ot":
            losses.append(sinkhorn_ot_loss(syn, real, ot_epsilon, ot_iters))
        elif metric == "cos_ot":
            losses.append(cosine_sinkhorn_ot_loss(syn, real, ot_epsilon, ot_iters))
        else:
            raise ValueError(f"Unknown latent metric: {metric}")
    return torch.stack(losses).mean()


def sample_class_batch(
    real: torch.Tensor,
    batch_size: int,
    hard_idx: torch.Tensor | None = None,
    hard_ratio: float = 0.0,
) -> torch.Tensor:
    if real.shape[0] <= batch_size:
        return real
    hard_idx = hard_idx if hard_idx is not None else torch.empty(0, dtype=torch.long, device=real.device)
    hard_ratio = min(max(float(hard_ratio), 0.0), 1.0)
    hard_count = int(round(batch_size * hard_ratio)) if hard_idx.numel() > 0 else 0
    hard_count = min(max(hard_count, 0), batch_size)
    random_count = batch_size - hard_count
    idx_parts = []
    if random_count > 0:
        idx_parts.append(torch.randint(0, real.shape[0], (random_count,), device=real.device))
    if hard_count > 0:
        hard_sample = torch.randint(0, hard_idx.numel(), (hard_count,), device=real.device)
        idx_parts.append(hard_idx[hard_sample])
    idx = torch.cat(idx_parts, dim=0)
    idx = idx[torch.randperm(idx.numel(), device=real.device)]
    return real[idx]


def mode_cover_schedule(
    base_weight: float,
    step: int,
    total_steps: int,
    schedule: str,
    start_frac: float,
    end_frac: float,
) -> float:
    if base_weight <= 0.0 or total_steps <= 0:
        return float(base_weight)
    if schedule == "none":
        return float(base_weight)
    start = min(max(float(start_frac), 0.0), 1.0)
    end = min(max(float(end_frac), 0.0), 1.0)
    if end <= start:
        raise ValueError("--mode-cover-anneal-end must be larger than --mode-cover-anneal-start")
    step_frac = 0.0 if total_steps <= 1 else (step - 1) / float(total_steps - 1)
    progress = min(max((step_frac - start) / (end - start), 0.0), 1.0)
    if schedule == "linear":
        factor = 1.0 - progress
    elif schedule == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unknown mode cover annealing schedule: {schedule}")
    return float(base_weight) * factor


def sinkhorn_assignment(
    cost: torch.Tensor,
    mode_prior: torch.Tensor,
    epsilon: float,
    iters: int,
) -> torch.Tensor:
    """Balanced soft prototype-to-mode assignment.

    The transport plan follows entropic OT (Cuturi, 2013): rows are synthetic
    prototype mass and columns are mode budget mass. The returned assignment is
    row-normalized so each prototype has a soft distribution over mode slots.
    """

    n = cost.shape[0]
    row_mass = torch.full((n,), 1.0 / max(n, 1), device=cost.device, dtype=cost.dtype)
    col_mass = mode_prior.to(device=cost.device, dtype=cost.dtype)
    col_mass = col_mass / col_mass.sum().clamp_min(1e-8)
    log_a = torch.log(row_mass.clamp_min(1e-8))
    log_b = torch.log(col_mass.clamp_min(1e-8))
    log_k = -cost / max(float(epsilon), 1e-8)
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(max(int(iters), 1)):
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_k + u.unsqueeze(1), dim=0)
    plan = torch.exp(log_k + u.unsqueeze(1) + v.unsqueeze(0))
    return plan / row_mass.unsqueeze(1).clamp_min(1e-8)


def soft_mode_cover_loss(
    syn_emb: torch.Tensor,
    mode_targets: torch.Tensor,
    mode_prior: torch.Tensor,
    strategy: str,
    temperature: float,
    sinkhorn_iters: int,
    balance_weight: float,
) -> torch.Tensor:
    mode_prior = mode_prior.to(device=syn_emb.device, dtype=syn_emb.dtype)
    mode_prior = mode_prior / mode_prior.sum().clamp_min(1e-8)
    cost = pairwise_sqdist(syn_emb, mode_targets) / max(float(syn_emb.shape[1]), 1.0)
    if strategy == "softmax":
        assignment = torch.softmax(-cost / max(float(temperature), 1e-8), dim=1)
        match_loss = (assignment * cost).sum(dim=1).mean()
        mass = assignment.mean(dim=0)
        balance_loss = F.mse_loss(mass, mode_prior)
        return match_loss + float(balance_weight) * balance_loss
    if strategy == "sinkhorn":
        assignment = sinkhorn_assignment(cost, mode_prior, temperature, sinkhorn_iters)
        return (assignment * cost).sum(dim=1).mean()
    raise ValueError(f"Unknown soft mode cover strategy: {strategy}")


def latent_dm_loss(
    z_syn: torch.Tensor,
    y_syn: torch.Tensor,
    mode_syn: torch.Tensor | None,
    stats: list[dict],
    num_classes: int,
    width: int,
    feature_dim: int,
    real_batch: int,
    normalize: bool,
    quantile_weight: float = 0.0,
    quantiles: torch.Tensor | None = None,
    difficulty_ratio: float = 0.0,
    mode_cover_weight: float = 0.0,
    mode_cover_strategy: str = "hard",
    mode_cover_temperature: float = 0.1,
    mode_cover_sinkhorn_iters: int = 20,
    mode_cover_balance_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedder = LatentEmbedder3D(z_syn.shape[1], width=width, feature_dim=feature_dim).to(z_syn.device)
    for parameter in embedder.parameters():
        parameter.requires_grad_(False)
    embedder.train()
    mean_losses = []
    quantile_losses = []
    for c in range(num_classes):
        syn = z_syn[y_syn == c]
        real = sample_class_batch(
            stats[c]["real_tensor"],
            real_batch,
            stats[c].get("hard_idx"),
            difficulty_ratio,
        )
        syn_emb = embedder(syn)
        real_emb = embedder(real)
        if normalize:
            syn_emb = F.normalize(syn_emb, dim=1)
            real_emb = F.normalize(real_emb, dim=1)
        mean_losses.append(F.mse_loss(syn_emb.mean(dim=0), real_emb.mean(dim=0)))
        if quantile_weight > 0.0:
            if quantiles is None:
                raise ValueError("quantiles must be provided when quantile_weight > 0")
            syn_q = torch.quantile(syn_emb, quantiles, dim=0)
            real_q = torch.quantile(real_emb.detach(), quantiles, dim=0)
            quantile_losses.append(F.mse_loss(syn_q, real_q))
        if mode_cover_weight > 0.0:
            if mode_syn is None:
                raise ValueError("mode_syn must be provided when mode_cover_weight > 0")
            num_modes = int(stats[c]["num_modes"])
            if num_modes <= 0:
                raise ValueError("mode_cover_weight requires --mode-cover-k > 0")
            if mode_cover_strategy == "hard":
                class_modes = mode_syn[y_syn == c]
                real_modes = stats[c]["mode_assignments"]
                for mode_id in range(num_modes):
                    syn_mode = syn[class_modes == mode_id]
                    if syn_mode.numel() == 0:
                        continue
                    real_mode_tensor = stats[c]["real_tensor"][real_modes == mode_id]
                    if real_mode_tensor.numel() == 0:
                        continue
                    real_mode = sample_class_batch(real_mode_tensor, real_batch, None, 0.0)
                    syn_mode_emb = embedder(syn_mode)
                    real_mode_emb = embedder(real_mode)
                    if normalize:
                        syn_mode_emb = F.normalize(syn_mode_emb, dim=1)
                        real_mode_emb = F.normalize(real_mode_emb, dim=1)
                    mean_losses.append(mode_cover_weight * F.mse_loss(syn_mode_emb.mean(dim=0), real_mode_emb.mean(dim=0)))
            else:
                mode_centers = stats[c]["mode_centers"].reshape(num_modes, *syn.shape[1:])
                mode_target_emb = embedder(mode_centers)
                if normalize:
                    mode_target_emb = F.normalize(mode_target_emb, dim=1)
                mean_losses.append(
                    mode_cover_weight
                    * soft_mode_cover_loss(
                        syn_emb,
                        mode_target_emb,
                        stats[c]["mode_prior"],
                        mode_cover_strategy,
                        mode_cover_temperature,
                        mode_cover_sinkhorn_iters,
                        mode_cover_balance_weight,
                    )
                )
    mean_loss = torch.stack(mean_losses).sum()
    quantile_loss = (
        torch.stack(quantile_losses).sum()
        if quantile_losses
        else z_syn.new_tensor(0.0)
    )
    return mean_loss, quantile_loss


def latent_gradient_matching_loss(
    z_syn: torch.Tensor,
    y_syn: torch.Tensor,
    stats: list[dict],
    num_classes: int,
    model_name: str,
    width: int,
    real_batch: int,
    dis_metric: str,
    bn_calib_per_class: int,
) -> torch.Tensor:
    """Official DC-style gradient matching, applied to autoencoder latents."""

    device = z_syn.device
    model = build_model(model_name, in_channels=z_syn.shape[1], num_classes=num_classes, width=width).to(device)
    model.train()
    if bn_calib_per_class > 0:
        real_tensors = []
        real_labels = []
        for c in range(num_classes):
            real_c = sample_class_batch(stats[c]["real_tensor"], bn_calib_per_class, None, 0.0)
            real_tensors.append(real_c)
            real_labels.append(torch.full((real_c.shape[0],), c, dtype=torch.long, device=device))
        calib_x = torch.cat(real_tensors, dim=0)
        calib_y = torch.cat(real_labels, dim=0)
        calib_indices = class_indices(calib_y, num_classes)
        calibrate_batchnorm_stats(model, calib_x, calib_indices, bn_calib_per_class)
    freeze_batchnorm_stats(model)

    criterion = nn.CrossEntropyLoss()
    losses = []
    for c in range(num_classes):
        syn = z_syn[y_syn == c]
        real = sample_class_batch(stats[c]["real_tensor"], real_batch, None, 0.0)
        y_real = torch.full((real.shape[0],), c, dtype=torch.long, device=device)
        y_syn_c = torch.full((syn.shape[0],), c, dtype=torch.long, device=device)
        real_loss = criterion(model(real), y_real)
        gw_real = torch.autograd.grad(real_loss, model.parameters(), retain_graph=False)
        gw_real = [g.detach() for g in gw_real]
        syn_loss = criterion(model(syn), y_syn_c)
        gw_syn = torch.autograd.grad(syn_loss, model.parameters(), create_graph=True)
        losses.append(match_gradients(list(gw_syn), gw_real, dis_metric))
    return torch.stack(losses).sum()


def parse_quantiles(text: str, device: torch.device) -> torch.Tensor:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("--dm-quantiles must contain at least one value")
    for value in values:
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"Quantile values must be in (0, 1), got {value}")
    return torch.tensor(values, dtype=torch.float32, device=device)


def weighted_token_stats(tokens: torch.Tensor, mode: str, power: float) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "uniform":
        return tokens.mean(dim=0), tokens.std(dim=0, unbiased=False)
    if mode != "activation":
        raise ValueError(f"Unknown local token weight mode: {mode}")
    weights = tokens.detach().norm(dim=1).clamp_min(1e-8).pow(power)
    weights = weights / weights.sum().clamp_min(1e-8)
    mean = (weights[:, None] * tokens).sum(dim=0)
    var = (weights[:, None] * (tokens - mean).pow(2)).sum(dim=0)
    return mean, var.clamp_min(1e-12).sqrt()


def local_token_moment_loss(
    z_syn: torch.Tensor,
    y_syn: torch.Tensor,
    stats: list[dict],
    num_classes: int,
    std_weight: float,
    weight_mode: str,
    weight_power: float,
) -> torch.Tensor:
    """Match class-wise local latent-token statistics instead of only whole-volume latents."""

    losses = []
    for c in range(num_classes):
        syn = z_syn[y_syn == c]
        tokens = syn.flatten(2).transpose(1, 2).reshape(-1, syn.shape[1])
        mean, std = weighted_token_stats(tokens, weight_mode, weight_power)
        mean_key = "real_token_weighted_mean" if weight_mode != "uniform" else "real_token_mean"
        std_key = "real_token_weighted_std" if weight_mode != "uniform" else "real_token_std"
        mean_loss = F.mse_loss(mean, stats[c][mean_key])
        std_loss = F.mse_loss(std, stats[c][std_key])
        losses.append(mean_loss + std_weight * std_loss)
    return torch.stack(losses).mean()


def classwise_pairwise_cosine_distance(z: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    values = []
    flat = F.normalize(z.flatten(1), dim=1)
    for c in range(num_classes):
        zc = flat[y == c]
        if zc.shape[0] < 2:
            continue
        dist = 1.0 - zc @ zc.T
        mask = ~torch.eye(zc.shape[0], dtype=torch.bool, device=zc.device)
        values.append(dist[mask].mean())
    if not values:
        return z.new_tensor(0.0)
    return torch.stack(values).mean()


def latent_diversity_floor_loss(
    z_syn: torch.Tensor,
    y_syn: torch.Tensor,
    num_classes: int,
    target: torch.Tensor,
) -> torch.Tensor:
    current = classwise_pairwise_cosine_distance(z_syn, y_syn, num_classes)
    return F.relu(target - current)


def decoder_cycle_loss(
    ae: ConvAutoencoder3D,
    z_syn: torch.Tensor,
    output_shape: tuple[int, int, int],
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0 or batch_size >= z_syn.shape[0]:
        z_batch = z_syn
    else:
        idx = torch.randperm(z_syn.shape[0], device=z_syn.device)[:batch_size]
        z_batch = z_syn[idx]
    x_cycle = ae.decode(z_batch, output_shape=output_shape).clamp(0.0, 1.0)
    z_cycle = ae.encode(x_cycle)
    return F.mse_loss(z_cycle, z_batch)


def run_latent_dd(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_total = time.perf_counter()

    info = get_dataset_info(args.dataset, args.data_root)
    task = infer_task(info)
    num_classes = len(info["label"])
    train_ds = load_medmnist_dataset(args.dataset, "train", args.data_root)
    val_ds = load_medmnist_dataset(args.dataset, "val", args.data_root)
    test_ds = load_medmnist_dataset(args.dataset, "test", args.data_root)
    x_train, y_train = prepare_tensors(train_ds)
    x_val, y_val = prepare_tensors(val_ds)
    x_test, y_test = prepare_tensors(test_ds)
    if args.limit_train_samples is not None:
        x_train = x_train[: args.limit_train_samples]
        y_train = y_train[: args.limit_train_samples]
    if args.limit_val_samples is not None:
        x_val = x_val[: args.limit_val_samples]
        y_val = y_val[: args.limit_val_samples]
    if args.limit_test_samples is not None:
        x_test = x_test[: args.limit_test_samples]
        y_test = y_test[: args.limit_test_samples]
    val_loader = make_loader(x_val, y_val, args.eval_batch_size, False, args.num_workers)
    test_loader = make_loader(x_test, y_test, args.eval_batch_size, False, args.num_workers)

    ae, ae_payload = load_autoencoder(args.autoencoder_checkpoint, device)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    storage_ratio = latent_storage_ratio(ae_payload)
    z_real = encode_volume_bank(ae, x_train, args.encode_batch_size, device)
    real_mode_assignments = None
    mode_center_overrides = None
    if args.mode_cover_k > 0:
        z_init_cpu, y_syn, mode_syn, real_mode_assignments, mode_center_overrides = init_mode_covered_latents(
            z_real,
            y_train,
            num_classes,
            args.ipc,
            args.lpc_multiplier,
            args.seed,
            args.mode_cover_k,
            args.mode_cover_allocation,
            args.mode_cover_anchor_init,
            args.mode_cover_prototype_init,
        )
        if args.mode_cover_assignment == "shuffled":
            real_mode_assignments = shuffle_mode_assignments(
                real_mode_assignments,
                args.mode_cover_k,
                args.seed,
            )
    else:
        z_init_cpu, y_syn = init_latents(
            z_real,
            y_train,
            num_classes,
            args.ipc,
            args.lpc_multiplier,
            args.seed,
            args.latent_init,
            args.mixed_random_frac,
        )
        mode_syn = torch.full_like(y_syn, -1)
    decoded_ipc = args.ipc * args.lpc_multiplier
    storage_budget_ipc = decoded_ipc / storage_ratio if storage_ratio == storage_ratio else float("nan")
    train_class_counts = [int((y_train == c).sum().item()) for c in range(num_classes)]
    replacement_classes = [
        c for c, count in enumerate(train_class_counts) if args.latent_init == "random" and count < decoded_ipc
    ]
    z_init = z_init_cpu.to(device)
    y_syn_device = y_syn.to(device)
    mode_syn_device = mode_syn.to(device)
    z_syn = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z_syn], lr=args.latent_lr)
    stats = prepare_latent_stats(
        z_real,
        y_train,
        num_classes,
        device,
        args.latent_metric,
        args.local_token_weight_mode,
        args.local_token_weight_power,
        args.dm_difficulty_mode,
        args.dm_difficulty_hard_quantile,
        args.mode_cover_k,
        args.mode_cover_allocation,
        real_mode_assignments,
        mode_center_overrides,
    )
    diversity_target = classwise_pairwise_cosine_distance(z_init, y_syn_device, num_classes).detach() * args.diversity_target_ratio
    dm_quantiles = parse_quantiles(args.dm_quantiles, device)
    if args.latent_metric == "dm" and args.latent_init == "kmeans":
        print("[note] latent_metric=dm follows LatentDM-style random-network mean matching; KMeans init is kept as a controlled initialization.")

    start_distill = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_loss = 0.0 if args.latent_steps == 0 else float("inf")
    best_z = z_syn.detach().cpu().clone()
    for step in tqdm(range(1, args.latent_steps + 1), desc="latent-dd", leave=False, disable=args.no_progress):
        optimizer.zero_grad(set_to_none=True)
        quantile_loss = z_syn.new_tensor(0.0)
        cycle_loss = z_syn.new_tensor(0.0)
        current_mode_cover_weight = mode_cover_schedule(
            args.mode_cover_weight,
            step,
            args.latent_steps,
            args.mode_cover_anneal,
            args.mode_cover_anneal_start,
            args.mode_cover_anneal_end,
        )
        if args.latent_metric == "dm":
            dist_loss, quantile_loss = latent_dm_loss(
                z_syn,
                y_syn_device,
                mode_syn_device,
                stats,
                num_classes,
                args.dm_width,
                args.dm_feature_dim,
                args.dm_real_batch,
                args.dm_feature_normalize,
                args.dm_quantile_weight,
                dm_quantiles,
                args.dm_difficulty_ratio,
                current_mode_cover_weight,
                args.mode_cover_strategy,
                args.mode_cover_temperature,
                args.mode_cover_sinkhorn_iters,
                args.mode_cover_balance_weight,
            )
        elif args.latent_metric == "gm":
            dist_loss = latent_gradient_matching_loss(
                z_syn,
                y_syn_device,
                stats,
                num_classes,
                args.gm_model,
                args.gm_width,
                args.gm_real_batch,
                args.gm_dis_metric,
                args.gm_bn_calib_per_class,
            )
        else:
            dist_loss = latent_distribution_loss(
                z_syn,
                y_syn_device,
                stats,
                num_classes,
                args.latent_metric,
                args.ot_epsilon,
                args.ot_iters,
            )
        local_loss = z_syn.new_tensor(0.0)
        if args.local_token_weight > 0.0:
            local_loss = local_token_moment_loss(
                z_syn,
                y_syn_device,
                stats,
                num_classes,
                args.local_token_std_weight,
                args.local_token_weight_mode,
                args.local_token_weight_power,
            )
        diversity_loss = z_syn.new_tensor(0.0)
        if args.diversity_weight > 0.0:
            diversity_loss = latent_diversity_floor_loss(
                z_syn,
                y_syn_device,
                num_classes,
                diversity_target,
            )
        if args.decoder_cycle_weight > 0.0:
            cycle_loss = decoder_cycle_loss(
                ae,
                z_syn,
                tuple(x_train.shape[-3:]),
                args.decoder_cycle_batch,
            )
        anchor = F.mse_loss(z_syn, z_init)
        loss = (
            args.dist_weight * (dist_loss + args.dm_quantile_weight * quantile_loss)
            + args.local_token_weight * local_loss
            + args.diversity_weight * diversity_loss
            + args.decoder_cycle_weight * cycle_loss
            + args.anchor_weight * anchor
        )
        loss.backward()
        optimizer.step()
        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_z = z_syn.detach().cpu().clone()
        if step == 1 or step % args.log_every == 0 or step == args.latent_steps:
            print(
                f"[latent-dd] step={step} loss={float(loss.item()):.6f} "
                f"dist={float(dist_loss.item()):.6f} q={float(quantile_loss.item()):.6f} "
                f"mode_w={current_mode_cover_weight:.6f} "
                f"local={float(local_loss.item()):.6f} "
                f"div={float(diversity_loss.item()):.6f} cycle={float(cycle_loss.item()):.6f} "
                f"anchor={float(anchor.item()):.6f}"
            )
    distill_runtime = time.perf_counter() - start_distill
    distill_peak_memory_mb = 0.0
    if device.type == "cuda":
        distill_peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        torch.cuda.reset_peak_memory_stats(device)

    decode_export_start = time.perf_counter()
    with torch.no_grad():
        selected_z = z_syn.detach().cpu().clone() if args.latent_select == "final" else best_z
        z_final = selected_z.to(device)
        flat_final = z_final.flatten(1)
        flat_init = z_init.flatten(1)
        latent_delta_l2 = (flat_final - flat_init).norm(dim=1)
        init_l2 = flat_init.norm(dim=1).clamp_min(1e-8)
        latent_delta_rel = latent_delta_l2 / init_l2
        x_syn = ae.decode(z_final, output_shape=tuple(x_train.shape[-3:])).detach().cpu().clamp(0.0, 1.0)
    decode_export_runtime = time.perf_counter() - decode_export_start
    decode_export_peak_memory_mb = 0.0
    if device.type == "cuda":
        decode_export_peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        torch.cuda.reset_peak_memory_stats(device)
    eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    eval_metrics = {}
    if not args.skip_eval:
        seed_everything(eval_seed)
        eval_metrics = eval_synthetic(
            x_syn=x_syn,
            y_syn=y_syn,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            task=task,
            num_classes=num_classes,
            in_channels=x_train.shape[1],
        )
    total_runtime = time.perf_counter() - start_total
    eval_peak_memory_mb = 0.0
    if device.type == "cuda" and not args.skip_eval:
        eval_peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    method_slug = args.method_name.lower().replace(" ", "_").replace("/", "_")
    run_name = (
        f"{args.dataset}_{method_slug}_{args.latent_metric}_{args.latent_init}_"
        f"ipc{args.ipc}_lpcx{args.lpc_multiplier}_seed{args.seed}"
    )
    run_dir = args.output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = run_dir / "synthetic_decoded.pt"
    synthetic_png = run_dir / "synthetic_central_slices.png"
    save_synthetic_grid(x_syn, y_syn, synthetic_png)
    torch.save(
        {
            "x_syn": x_syn,
            "y_syn": y_syn,
            "mode_syn": mode_syn,
            "z_syn": best_z,
            "z_final": selected_z,
            "z_init": z_init_cpu,
            "dataset": args.dataset,
            "ipc": args.ipc,
            "lpc_multiplier": args.lpc_multiplier,
            "decoded_ipc": decoded_ipc,
            "latent_storage_ratio": storage_ratio,
            "storage_budget_ipc": storage_budget_ipc,
            "seed": args.seed,
            "autoencoder_checkpoint": str(args.autoencoder_checkpoint),
            "autoencoder_latent_shape": ae_payload.get("latent_shape"),
            "mode_cover_prototype_init": args.mode_cover_prototype_init,
        },
        synthetic_path,
    )
    row = {
        "dataset": args.dataset,
        "method": args.method_name,
        "distilled_object": "decoded_3d_volume_from_latent",
        "latent_metric": args.latent_metric,
        "latent_init": args.latent_init,
        "mixed_random_frac": args.mixed_random_frac,
        "ipc": args.ipc,
        "lpc_multiplier": args.lpc_multiplier,
        "mode_cover_k": args.mode_cover_k,
        "mode_cover_weight": args.mode_cover_weight,
        "mode_cover_strategy": args.mode_cover_strategy,
        "mode_cover_temperature": args.mode_cover_temperature,
        "mode_cover_sinkhorn_iters": args.mode_cover_sinkhorn_iters,
        "mode_cover_balance_weight": args.mode_cover_balance_weight,
        "mode_cover_anneal": args.mode_cover_anneal,
        "mode_cover_anneal_start": args.mode_cover_anneal_start,
        "mode_cover_anneal_end": args.mode_cover_anneal_end,
        "mode_cover_allocation": args.mode_cover_allocation,
        "mode_cover_assignment": args.mode_cover_assignment,
        "mode_cover_anchor_init": args.mode_cover_anchor_init,
        "mode_cover_prototype_init": args.mode_cover_prototype_init,
        "decoded_ipc": decoded_ipc,
        "num_synthetic_volumes": int(y_syn.numel()),
        "latent_storage_ratio": storage_ratio,
        "storage_budget_ipc": storage_budget_ipc,
        "seed": args.seed,
        "eval_seed": eval_seed,
        "eval_model": args.eval_model,
        "eval_width": args.eval_width,
        "eval_epochs": args.eval_epochs,
        "eval_batch_size": args.eval_batch_size,
        "train_class_counts": json.dumps(train_class_counts),
        "random_init_replacement_classes": json.dumps(replacement_classes),
        "status": "completed",
        "error_type": "",
        "eval_skipped": bool(args.skip_eval),
        "test_balanced_acc": eval_metrics.get("test_balanced_acc", float("nan")),
        "test_auc": eval_metrics.get("test_auc", float("nan")),
        "test_positive_recall": eval_metrics.get("test_recall_c1", float("nan")),
        "test_negative_recall": eval_metrics.get("test_recall_c0", float("nan")),
        "eval_best_epoch": eval_metrics.get("eval_best_epoch", -1),
        "distill_runtime_sec": distill_runtime,
        "distill_seconds_per_step": distill_runtime / max(1, args.latent_steps),
        "decode_export_runtime_sec": decode_export_runtime,
        "runtime_sec": total_runtime,
        "peak_memory_mb": distill_peak_memory_mb,
        "distill_peak_memory_mb": distill_peak_memory_mb,
        "decode_export_peak_memory_mb": decode_export_peak_memory_mb,
        "eval_peak_memory_mb": eval_peak_memory_mb,
        "best_latent_loss": best_loss,
        "latent_select": args.latent_select,
        "latent_lr": args.latent_lr,
        "latent_steps": args.latent_steps,
        "dist_weight": args.dist_weight,
        "anchor_weight": args.anchor_weight,
        "dm_quantile_weight": args.dm_quantile_weight,
        "dm_quantiles": args.dm_quantiles,
        "dm_difficulty_ratio": args.dm_difficulty_ratio,
        "dm_difficulty_mode": args.dm_difficulty_mode,
        "dm_difficulty_hard_quantile": args.dm_difficulty_hard_quantile,
        "gm_model": args.gm_model,
        "gm_width": args.gm_width,
        "gm_real_batch": args.gm_real_batch,
        "gm_dis_metric": args.gm_dis_metric,
        "gm_bn_calib_per_class": args.gm_bn_calib_per_class,
        "decoder_cycle_weight": args.decoder_cycle_weight,
        "decoder_cycle_batch": args.decoder_cycle_batch,
        "local_token_weight": args.local_token_weight,
        "local_token_std_weight": args.local_token_std_weight,
        "local_token_weight_mode": args.local_token_weight_mode,
        "local_token_weight_power": args.local_token_weight_power,
        "diversity_weight": args.diversity_weight,
        "diversity_target_ratio": args.diversity_target_ratio,
        "diversity_target": float(diversity_target.item()),
        "latent_delta_l2_mean": float(latent_delta_l2.mean().item()),
        "latent_delta_l2_max": float(latent_delta_l2.max().item()),
        "latent_delta_rel_mean": float(latent_delta_rel.mean().item()),
        "latent_delta_rel_max": float(latent_delta_rel.max().item()),
        "synthetic_path": str(synthetic_path),
        "synthetic_png": str(synthetic_png),
        "autoencoder_checkpoint": str(args.autoencoder_checkpoint),
        "command": " ".join(sys.argv),
    }
    for key, value in eval_metrics.items():
        row[key] = value
    append_dm_csv(args.output_dir / "metrics" / "latent_dd_3d_runs.csv", row)
    (run_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LatentDD-style 3D latent distribution matching baseline")
    parser.add_argument("--dataset", default="mosmed_binary96")
    parser.add_argument("--ipc", type=int, default=1)
    parser.add_argument("--lpc-multiplier", type=int, default=1, help="Capacity diagnostic: decoded volumes per class = ipc * lpc_multiplier.")
    parser.add_argument("--mode-cover-k", type=int, default=0, help="If positive, initialize and match synthetic latents across K class-wise latent modes.")
    parser.add_argument("--mode-cover-anchor-init", choices=["kmeans", "random"], default="kmeans", help="How class-wise mode anchors are initialized before assigning real latents.")
    parser.add_argument("--mode-cover-prototype-init", choices=["random", "medoid"], default="random", help="How prototypes are initialized within each mode-covered anchor cluster.")
    parser.add_argument("--mode-cover-weight", type=float, default=0.0, help="Additional weight for mode-wise LatentDM terms.")
    parser.add_argument("--mode-cover-allocation", choices=["proportional", "uniform"], default="proportional", help="How decoded IPC slots are allocated across class-wise latent modes.")
    parser.add_argument("--mode-cover-assignment", choices=["matched", "shuffled"], default="matched", help="Use matched mode targets or shuffled targets for assignment ablations.")
    parser.add_argument("--mode-cover-strategy", choices=["hard", "softmax", "sinkhorn"], default="hard", help="Mode coverage objective: hard fixed slots, softmax assignment, or entropic-OT/Sinkhorn assignment.")
    parser.add_argument("--mode-cover-temperature", type=float, default=0.1, help="Temperature/epsilon for softmax or Sinkhorn mode assignment.")
    parser.add_argument("--mode-cover-sinkhorn-iters", type=int, default=20, help="Sinkhorn-Knopp iterations for --mode-cover-strategy sinkhorn.")
    parser.add_argument("--mode-cover-balance-weight", type=float, default=1.0, help="Mass-prior penalty weight used by --mode-cover-strategy softmax.")
    parser.add_argument("--mode-cover-anneal", choices=["none", "linear", "cosine"], default="none", help="Anneal the mode coverage weight during latent optimization.")
    parser.add_argument("--mode-cover-anneal-start", type=float, default=0.0, help="Fraction of optimization steps where mode-cover annealing starts.")
    parser.add_argument("--mode-cover-anneal-end", type=float, default=1.0, help="Fraction of optimization steps where mode-cover annealing reaches zero.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-metric", choices=["ot", "cos_ot", "mmd", "dm", "gm"], default="ot")
    parser.add_argument("--latent-init", choices=["kmeans", "random", "mixed"], default="kmeans")
    parser.add_argument("--mixed-random-frac", type=float, default=0.5, help="Fraction of random real latents when --latent-init mixed.")
    parser.add_argument("--latent-steps", type=int, default=100)
    parser.add_argument("--latent-lr", type=float, default=1e-2)
    parser.add_argument("--latent-select", choices=["best_loss", "final"], default="best_loss")
    parser.add_argument("--dist-weight", type=float, default=1.0, help="Weight for the global latent distribution matching term.")
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-iters", type=int, default=30)
    parser.add_argument("--dm-width", type=int, default=64)
    parser.add_argument("--dm-feature-dim", type=int, default=128)
    parser.add_argument("--dm-real-batch", type=int, default=64)
    parser.add_argument("--dm-feature-normalize", action="store_true")
    parser.add_argument("--dm-quantile-weight", type=float, default=0.0, help="Weight of latent quantile matching inside the LatentDM term.")
    parser.add_argument("--dm-quantiles", default="0.1,0.25,0.5,0.75,0.9", help="Comma-separated quantiles for LQ-LatentDM.")
    parser.add_argument("--dm-difficulty-ratio", type=float, default=0.0, help="Fraction of each class-wise real latent batch sampled from hard latent examples.")
    parser.add_argument("--dm-difficulty-mode", choices=["none", "centroid_margin", "centroid_distance"], default="none")
    parser.add_argument("--dm-difficulty-hard-quantile", type=float, default=0.3, help="Top hard fraction per class used as the hard latent pool.")
    parser.add_argument("--gm-model", choices=["convnet3d", "resnet18_3d", "resnet10_3d_gn", "resnet18_3d_gn"], default="convnet3d", help="Network used for latent gradient matching.")
    parser.add_argument("--gm-width", type=int, default=16)
    parser.add_argument("--gm-real-batch", type=int, default=8)
    parser.add_argument("--gm-dis-metric", choices=["ours", "mse", "cos"], default="ours")
    parser.add_argument("--gm-bn-calib-per-class", type=int, default=8)
    parser.add_argument("--local-token-weight", type=float, default=0.0, help="Weight for class-wise spatial latent-token moment matching.")
    parser.add_argument("--local-token-std-weight", type=float, default=0.5, help="Std term weight inside local token moment matching.")
    parser.add_argument("--local-token-weight-mode", choices=["uniform", "activation"], default="uniform")
    parser.add_argument("--local-token-weight-power", type=float, default=1.0)
    parser.add_argument("--diversity-weight", type=float, default=0.0, help="Weight for preserving class-wise pairwise latent diversity.")
    parser.add_argument("--diversity-target-ratio", type=float, default=0.9, help="Diversity floor as a ratio of initialization diversity.")
    parser.add_argument("--decoder-cycle-weight", type=float, default=0.0, help="Weight for decoder-manifold cycle consistency E(D(z_syn)) ~= z_syn.")
    parser.add_argument("--decoder-cycle-batch", type=int, default=4, help="Synthetic latent batch size used by decoder cycle loss.")
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--method-name", default="LatentDD-3D")
    parser.add_argument("--eval-model", choices=["convnet3d", "resnet18_3d", "resnet10_3d_gn", "resnet18_3d_gn"], default="convnet3d")
    parser.add_argument("--eval-width", type=int, default=32)
    parser.add_argument("--eval-epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--skip-eval", action="store_true", help="Skip fresh evaluator training; useful for DD optimization-only resource profiling.")
    parser.add_argument("--eval-seed", type=int, default=None, help="Seed reset before training the fresh evaluator. Defaults to --seed.")
    parser.add_argument("--eval-optimizer", choices=["adamw", "sgd"], default="sgd")
    parser.add_argument("--eval-lr", type=float, default=0.01)
    parser.add_argument("--eval-momentum", type=float, default=0.9)
    parser.add_argument("--eval-weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-class-weight-mode", choices=["none", "inverse_freq"], default="none")
    parser.add_argument("--eval-balanced-sampler", action="store_true")
    parser.add_argument("--select-metric", choices=["auc_acc", "auc", "acc", "balanced_acc", "auc_bacc"], default="balanced_acc")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "medmnist")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "latent3d_dd_pilot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--limit-test-samples", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = run_latent_dd(args)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
