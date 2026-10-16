from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from models3d import build_model  # noqa: E402
from train_medmnist3d_cls import (  # noqa: E402
    DATASETS_3D,
    append_csv,
    compute_metrics,
    evaluate,
    get_dataset_info,
    infer_task,
    is_better,
    load_medmnist_dataset,
    make_loader,
    prepare_tensors,
    seed_everything,
    train_one_epoch,
)


def class_indices(y: torch.Tensor, num_classes: int) -> list[torch.Tensor]:
    return [torch.nonzero(y == c, as_tuple=False).flatten() for c in range(num_classes)]


def init_synthetic_from_real(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    ipc: int,
    seed: int,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    xs = []
    ys = []
    for c, indices in enumerate(class_indices(y_train, num_classes)):
        if indices.numel() < ipc:
            raise ValueError(f"Class {c} has only {indices.numel()} samples, cannot init ipc={ipc}")
        perm = torch.randperm(indices.numel(), generator=generator)[:ipc]
        selected = x_train[indices[perm]].clone()
        if noise_std > 0:
            selected = selected + torch.randn(selected.shape, generator=generator) * noise_std
        xs.append(selected.clamp(0.0, 1.0))
        ys.append(torch.full((ipc,), c, dtype=torch.long))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def append_dm_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            old_rows = []
            for old_row in reader:
                old_row.pop(None, None)
                old_rows.append(old_row)
        fieldnames = old_fields + [key for key in row.keys() if key not in old_fields]
        if fieldnames != old_fields:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(old_rows)
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def make_class_weights(y_train: torch.Tensor, num_classes: int, mode: str, tau: float) -> torch.Tensor:
    if mode == "none":
        return torch.ones(num_classes, device=y_train.device)
    counts = torch.bincount(y_train, minlength=num_classes).float()
    if mode == "inverse_freq":
        weights = counts.sum() / counts.clamp_min(1.0)
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")
    weights = weights.pow(tau)
    return weights / weights.mean().clamp_min(1e-12)


def normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    return weights / weights.mean().clamp_min(1e-12)


def class_weight_tensor(labels: torch.Tensor, num_classes: int, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = torch.bincount(labels.cpu(), minlength=num_classes).float()
    if mode == "inverse_freq":
        weights = counts.sum() / counts.clamp_min(1.0)
    else:
        raise ValueError(f"Unknown eval class weight mode: {mode}")
    return normalize_weights(weights).to(device)


def make_eval_train_loader(
    x_syn: torch.Tensor,
    y_syn: torch.Tensor,
    batch_size: int,
    workers: int,
    balanced_sampler: bool,
) -> DataLoader:
    ds = torch.utils.data.TensorDataset(x_syn.cpu(), y_syn.cpu())
    if not balanced_sampler:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    counts = torch.bincount(y_syn.cpu()).float()
    sample_weights = 1.0 / counts[y_syn.cpu()].clamp_min(1.0)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=True)


def eval_synthetic(
    x_syn: torch.Tensor,
    y_syn: torch.Tensor,
    val_loader: DataLoader,
    test_loader: DataLoader,
    args: argparse.Namespace,
    task: str,
    num_classes: int,
    in_channels: int,
) -> dict[str, float]:
    device = torch.device(args.device)
    model = build_model(args.eval_model, in_channels=in_channels, num_classes=num_classes, width=args.eval_width).to(device)
    ce_weight = class_weight_tensor(y_syn, num_classes, args.eval_class_weight_mode, device)
    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    if args.eval_optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.eval_lr,
            momentum=args.eval_momentum,
            weight_decay=args.eval_weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.eval_lr, weight_decay=args.eval_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.eval_epochs)
    train_loader = make_eval_train_loader(x_syn, y_syn, args.eval_batch_size, args.num_workers, args.eval_balanced_sampler)
    best_state = None
    best_val = {"auc": -1.0, "acc": -1.0, "balanced_acc": -1.0, "loss": float("inf")}
    best_epoch = -1
    train_runtime_sec = 0.0
    val_runtime_sec = 0.0
    for epoch in range(1, args.eval_epochs + 1):
        epoch_train_start = time.perf_counter()
        train_one_epoch(model, train_loader, criterion, optimizer, device, progress=False)
        train_runtime_sec += time.perf_counter() - epoch_train_start
        scheduler.step()
        if epoch % args.eval_every != 0 and epoch != args.eval_epochs:
            continue
        val_start = time.perf_counter()
        val_metrics = evaluate(model, val_loader, criterion, device, task, num_classes, progress=False)
        val_runtime_sec += time.perf_counter() - val_start
        if is_better(val_metrics, best_val, args.select_metric):
            best_val = dict(val_metrics)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_start = time.perf_counter()
    test_metrics = evaluate(model, test_loader, criterion, device, task, num_classes, progress=False)
    test_runtime_sec = time.perf_counter() - test_start
    metrics = {
        "eval_best_epoch": best_epoch,
        "eval_val_auc": best_val["auc"],
        "eval_val_acc": best_val["acc"],
        "eval_val_balanced_acc": best_val.get("balanced_acc", float("nan")),
        "test_auc": test_metrics["auc"],
        "test_acc": test_metrics["acc"],
        "test_balanced_acc": test_metrics.get("balanced_acc", float("nan")),
        "test_loss": test_metrics["loss"],
        "eval_train_runtime_sec": train_runtime_sec,
        "eval_val_runtime_sec": val_runtime_sec,
        "eval_test_runtime_sec": test_runtime_sec,
        "eval_measured_runtime_sec": train_runtime_sec + val_runtime_sec + test_runtime_sec,
        "eval_train_seconds_per_epoch": train_runtime_sec / max(1, args.eval_epochs),
    }
    for c in range(num_classes):
        metrics[f"eval_val_recall_c{c}"] = best_val.get(f"recall_c{c}", float("nan"))
        metrics[f"test_recall_c{c}"] = test_metrics.get(f"recall_c{c}", float("nan"))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-style 2D dataset condensation baselines adapted to 3D MedMNIST")
    parser.add_argument("--dataset", default="organmnist3d")
    parser.add_argument("--dd-method", choices=["dm", "dc"], default="dm", help="DM matches feature distributions; DC matches one-step gradients.")
    parser.add_argument("--ipc", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-real", type=int, default=64)
    parser.add_argument("--lr-syn", type=float, default=0.1)
    parser.add_argument("--init-noise-std", type=float, default=0.01)
    parser.add_argument("--dm-model", choices=["convnet3d", "resnet18_3d", "resnet10_3d_gn", "resnet18_3d_gn"], default="convnet3d")
    parser.add_argument("--dm-width", type=int, default=32)
    parser.add_argument("--eval-model", choices=["convnet3d", "resnet18_3d", "resnet10_3d_gn", "resnet18_3d_gn"], default="convnet3d")
    parser.add_argument("--eval-width", type=int, default=32)
    parser.add_argument("--eval-epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--eval-lr", type=float, default=1e-3)
    parser.add_argument("--eval-momentum", type=float, default=0.9)
    parser.add_argument("--eval-weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-class-weight-mode", choices=["none", "inverse_freq"], default="none")
    parser.add_argument("--eval-balanced-sampler", action="store_true")
    parser.add_argument("--select-metric", choices=["auc_acc", "auc", "acc", "balanced_acc", "auc_bacc"], default="auc_acc")
    parser.add_argument("--method-name", default="3d_dm")
    parser.add_argument("--class-weight-mode", choices=["none", "inverse_freq"], default="none")
    parser.add_argument("--class-weight-tau", type=float, default=1.0)
    parser.add_argument("--adaptive-weight-mode", choices=["none", "residual_ema"], default="none")
    parser.add_argument("--adaptive-weight-gamma", type=float, default=1.0)
    parser.add_argument("--adaptive-weight-momentum", type=float, default=0.9)
    parser.add_argument("--feature-normalize", action="store_true")
    parser.add_argument("--prototype-margin-weight", type=float, default=0.0)
    parser.add_argument("--prototype-margin", type=float, default=0.5)
    parser.add_argument("--dis-metric", choices=["ours", "cos", "mse"], default="ours")
    parser.add_argument("--syn-optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--syn-momentum", type=float, default=0.5)
    parser.add_argument("--bn-calib-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "medmnist")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def distance_wb_nd(syn_grad: torch.Tensor, real_grad: torch.Tensor) -> torch.Tensor:
    """Official DC-style layer/filter-wise cosine distance, extended to 3D tensors."""
    if syn_grad.ndim <= 1:
        return torch.tensor(0.0, device=syn_grad.device)
    syn_view = syn_grad.reshape(syn_grad.shape[0], -1)
    real_view = real_grad.reshape(real_grad.shape[0], -1)
    numerator = (syn_view * real_view).sum(dim=-1)
    denominator = syn_view.norm(dim=-1) * real_view.norm(dim=-1) + 1e-6
    return (1.0 - numerator / denominator).sum()


def match_gradients(gw_syn: list[torch.Tensor], gw_real: list[torch.Tensor], metric: str) -> torch.Tensor:
    loss = torch.tensor(0.0, device=gw_syn[0].device)
    for syn_grad, real_grad in zip(gw_syn, gw_real):
        if metric == "ours":
            loss = loss + distance_wb_nd(syn_grad, real_grad)
        elif metric == "mse":
            syn_flat = syn_grad.reshape(-1)
            real_flat = real_grad.reshape(-1)
            loss = loss + F.mse_loss(syn_flat, real_flat)
        else:
            syn_flat = syn_grad.reshape(-1)
            real_flat = real_grad.reshape(-1)
            loss = loss + (1.0 - F.cosine_similarity(syn_flat, real_flat, dim=0, eps=1e-8))
    return loss


def has_batchnorm(model: nn.Module) -> bool:
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    return any(isinstance(module, bn_types) for module in model.modules())


def freeze_batchnorm_stats(model: nn.Module) -> None:
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for module in model.modules():
        if isinstance(module, bn_types):
            module.eval()


def calibrate_batchnorm_stats(
    model: nn.Module,
    x_train: torch.Tensor,
    cls_indices: list[torch.Tensor],
    per_class: int,
) -> None:
    if per_class <= 0 or not has_batchnorm(model):
        return
    real_batches = []
    for indices in cls_indices:
        if indices.numel() == 0:
            continue
        count = min(per_class, indices.numel())
        pick = indices[torch.randint(indices.numel(), (count,), device=x_train.device)]
        real_batches.append(x_train[pick])
    if not real_batches:
        return
    model.train()
    with torch.no_grad():
        model(torch.cat(real_batches, dim=0))
    freeze_batchnorm_stats(model)


def main() -> None:
    args = parse_args()
    if args.method_name == "3d_dm" and args.dd_method != "dm":
        args.method_name = f"3d_{args.dd_method}"
    seed_everything(args.seed)
    info = get_dataset_info(args.dataset, args.data_root)
    task = infer_task(info)
    num_classes = len(info["label"])
    train_ds = load_medmnist_dataset(args.dataset, "train", args.data_root)
    val_ds = load_medmnist_dataset(args.dataset, "val", args.data_root)
    test_ds = load_medmnist_dataset(args.dataset, "test", args.data_root)
    x_train, y_train = prepare_tensors(train_ds)
    x_val, y_val = prepare_tensors(val_ds)
    x_test, y_test = prepare_tensors(test_ds)
    in_channels = x_train.shape[1]

    device = torch.device(args.device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    cls_indices = class_indices(y_train, num_classes)
    x_syn_init, y_syn = init_synthetic_from_real(
        x_train.cpu(),
        y_train.cpu(),
        num_classes,
        args.ipc,
        args.seed,
        args.init_noise_std,
    )
    x_syn = nn.Parameter(x_syn_init.to(device))
    y_syn = y_syn.to(device)
    if args.syn_optimizer == "sgd":
        optimizer = torch.optim.SGD([x_syn], lr=args.lr_syn, momentum=args.syn_momentum)
    else:
        optimizer = torch.optim.Adam([x_syn], lr=args.lr_syn)
    criterion_dc = nn.CrossEntropyLoss()
    base_class_weights = make_class_weights(y_train, num_classes, args.class_weight_mode, args.class_weight_tau)
    class_weights = base_class_weights.clone()
    residual_ema = torch.ones(num_classes, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_start_time = time.perf_counter()
    distill_start_time = time.perf_counter()

    for it in range(1, args.iterations + 1):
        model = build_model(args.dm_model, in_channels=in_channels, num_classes=num_classes, width=args.dm_width).to(device)
        model.train()
        calibrate_batchnorm_stats(model, x_train, cls_indices, args.bn_calib_per_class)
        if args.dd_method == "dm":
            for p in model.parameters():
                p.requires_grad_(False)
        else:
            for p in model.parameters():
                p.requires_grad_(True)
        freeze_batchnorm_stats(model)

        loss = torch.tensor(0.0, device=device)
        current_residuals = torch.zeros(num_classes, device=device)
        syn_prototypes = []
        for c, indices in enumerate(cls_indices):
            pick = indices[torch.randint(indices.numel(), (min(args.batch_real, indices.numel()),), device=device)]
            real = x_train[pick]
            syn = x_syn[c * args.ipc : (c + 1) * args.ipc]
            if args.dd_method == "dm":
                real_feat = model.forward_features(real).detach()
                syn_feat = model.forward_features(syn)
                if args.feature_normalize:
                    real_feat = F.normalize(real_feat, dim=1)
                    syn_feat = F.normalize(syn_feat, dim=1)
                class_loss = F.mse_loss(syn_feat.mean(dim=0), real_feat.mean(dim=0))
                syn_prototypes.append(syn_feat.mean(dim=0))
            else:
                y_real = torch.full((real.shape[0],), c, dtype=torch.long, device=device)
                y_syn_c = torch.full((syn.shape[0],), c, dtype=torch.long, device=device)
                real_logits = model(real)
                real_loss = criterion_dc(real_logits, y_real)
                gw_real = torch.autograd.grad(real_loss, model.parameters(), retain_graph=False)
                gw_real = [g.detach() for g in gw_real]
                syn_logits = model(syn)
                syn_loss = criterion_dc(syn_logits, y_syn_c)
                gw_syn = torch.autograd.grad(syn_loss, model.parameters(), create_graph=True)
                class_loss = match_gradients(list(gw_syn), gw_real, args.dis_metric)
                syn_prototypes.append(model.forward_features(syn).mean(dim=0))
            current_residuals[c] = class_loss.detach()
            loss = loss + class_weights[c] * class_loss

        if args.prototype_margin_weight > 0 and num_classes > 1:
            proto = torch.stack(syn_prototypes, dim=0)
            proto = F.normalize(proto, dim=1)
            distances = torch.pdist(proto, p=2)
            margin_loss = F.relu(args.prototype_margin - distances).pow(2).mean()
            loss = loss + args.prototype_margin_weight * margin_loss

        if args.adaptive_weight_mode == "residual_ema":
            with torch.no_grad():
                residual_ema.mul_(args.adaptive_weight_momentum).add_(
                    current_residuals, alpha=1.0 - args.adaptive_weight_momentum
                )
                residual_weights = normalize_weights(residual_ema.clamp_min(1e-12).pow(args.adaptive_weight_gamma))
                class_weights = normalize_weights(base_class_weights * residual_weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x_syn.clamp_(0.0, 1.0)
        if it == 1 or it % args.log_every == 0 or it == args.iterations:
            weight_str = ",".join(f"{float(w):.3f}" for w in class_weights.detach().cpu())
            print(f"[{args.dataset}] {args.dd_method}_iter={it:04d}/{args.iterations} loss={loss.item():.8e} weights=[{weight_str}]")

    distill_seconds = time.perf_counter() - distill_start_time
    peak_memory_mb = float("nan")
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    distill_peak_memory_mb = round(peak_memory_mb, 3) if not np.isnan(peak_memory_mb) else peak_memory_mb

    val_loader = make_loader(x_val, y_val, args.eval_batch_size, False, args.num_workers)
    test_loader = make_loader(x_test, y_test, args.eval_batch_size, False, args.num_workers)
    eval_start_time = time.perf_counter()
    eval_metrics = eval_synthetic(
        x_syn.detach(),
        y_syn.detach(),
        val_loader,
        test_loader,
        args,
        task,
        num_classes,
        in_channels,
    )
    eval_seconds = time.perf_counter() - eval_start_time
    total_seconds = time.perf_counter() - total_start_time

    method_tag = args.method_name
    if args.class_weight_mode != "none":
        method_tag = f"{method_tag}_{args.class_weight_mode}_tau{args.class_weight_tau:g}"
    if args.adaptive_weight_mode != "none":
        method_tag = f"{method_tag}_{args.adaptive_weight_mode}_g{args.adaptive_weight_gamma:g}"
    if args.feature_normalize:
        method_tag = f"{method_tag}_fnorm"
    if args.prototype_margin_weight > 0:
        method_tag = f"{method_tag}_pm{args.prototype_margin:g}w{args.prototype_margin_weight:g}"
    if args.eval_class_weight_mode != "none":
        method_tag = f"{method_tag}_evalcw"
    if args.eval_balanced_sampler:
        method_tag = f"{method_tag}_evalbs"
    run_name = f"{args.dataset}_{method_tag}_ipc{args.ipc}_{args.dm_model}_w{args.dm_width}_seed{args.seed}_{args.select_metric}"
    run_dir = args.output_dir / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    syn_path = run_dir / "synthetic.pt"
    torch.save({"images": x_syn.detach().cpu(), "labels": y_syn.detach().cpu(), "args": vars(args)}, syn_path)
    row = {
        "dataset": args.dataset,
        "method": args.method_name,
        "ipc": args.ipc,
        "seed": args.seed,
        "iterations": args.iterations,
        "dd_method": args.dd_method,
        "class_weight_mode": args.class_weight_mode,
        "class_weight_tau": args.class_weight_tau,
        "adaptive_weight_mode": args.adaptive_weight_mode,
        "adaptive_weight_gamma": args.adaptive_weight_gamma,
        "adaptive_weight_momentum": args.adaptive_weight_momentum,
        "feature_normalize": args.feature_normalize,
        "prototype_margin_weight": args.prototype_margin_weight,
        "prototype_margin": args.prototype_margin,
        "dis_metric": args.dis_metric,
        "syn_optimizer": args.syn_optimizer,
        "syn_momentum": args.syn_momentum,
        "bn_calib_per_class": args.bn_calib_per_class,
        "eval_optimizer": args.eval_optimizer,
        "eval_class_weight_mode": args.eval_class_weight_mode,
        "eval_balanced_sampler": args.eval_balanced_sampler,
        "dm_model": args.dm_model,
        "dm_width": args.dm_width,
        "eval_model": args.eval_model,
        "eval_width": args.eval_width,
        "eval_epochs": args.eval_epochs,
        "select_metric": args.select_metric,
        "train_size": int(y_syn.numel()),
        "full_train_size": len(train_ds),
        "synthetic_path": str(syn_path),
        "distilled_object": "synthetic_3d_volume",
        "requires_inner_training": False,
        "requires_second_order_grad": args.dd_method == "dc",
        "distill_seconds": round(distill_seconds, 3),
        "distill_runtime_sec": round(distill_seconds, 3),
        "distill_seconds_per_iteration": round(distill_seconds / max(1, args.iterations), 6),
        "eval_seconds": round(eval_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "peak_memory_mb": distill_peak_memory_mb,
        "distill_peak_memory_mb": distill_peak_memory_mb,
        "status": "ok",
        **{k: round(v, 6) if isinstance(v, float) else v for k, v in eval_metrics.items()},
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    metrics_name = "medmnist3d_2d_to3d_runs.csv" if args.method_name.startswith("3d_") else "medmnist3d_dm_runs.csv"
    append_dm_csv(args.output_dir / "metrics" / metrics_name, row)
    print(f"TEST {json.dumps(row, indent=2)}")


if __name__ == "__main__":
    main()
