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
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from models3d import build_model  # noqa: E402
from distill_3d_dm import eval_synthetic, init_synthetic_from_real  # noqa: E402
from train_medmnist3d_cls import (  # noqa: E402
    DATASETS_3D,
    get_dataset_info,
    infer_task,
    load_medmnist_dataset,
    make_loader,
    prepare_tensors,
    seed_everything,
)


def append_csv(path: Path, row: dict) -> None:
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


def flatten_named(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params.values()])


def unflatten_named(flat: torch.Tensor, template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    offset = 0
    for name, value in template.items():
        n = value.numel()
        out[name] = flat[offset : offset + n].view_as(value)
        offset += n
    return out


def sample_class_balanced(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    per_class: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for c in range(num_classes):
        idx = torch.nonzero(y == c, as_tuple=False).flatten()
        pick = idx[torch.randint(idx.numel(), (min(per_class, idx.numel()),), device=device)]
        xs.append(x[pick])
        ys.append(torch.full((pick.numel(),), c, dtype=torch.long, device=device))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def train_expert_trajectories(
    args: argparse.Namespace,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    in_channels: int,
    num_classes: int,
    device: torch.device,
) -> tuple[list[list[dict[str, torch.Tensor]]], float, float]:
    start = time.perf_counter()
    trajectories: list[list[dict[str, torch.Tensor]]] = []
    for expert_id in range(args.num_experts):
        model = build_model(args.mtt_model, in_channels=in_channels, num_classes=num_classes, width=args.mtt_width).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.expert_lr, momentum=args.expert_momentum, weight_decay=args.expert_weight_decay)
        trajectory = [{k: v.detach().cpu().clone() for k, v in model.named_parameters()}]
        for epoch in range(1, args.expert_epochs + 1):
            model.train()
            steps = max(1, args.expert_steps_per_epoch)
            for _ in range(steps):
                xb, yb = sample_class_balanced(x_train, y_train, num_classes, args.expert_batch_per_class, device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                optimizer.step()
            trajectory.append({k: v.detach().cpu().clone() for k, v in model.named_parameters()})
            print(f"[expert {expert_id}] epoch={epoch}/{args.expert_epochs} loss={float(loss.item()):.6f}")
        trajectories.append(trajectory)
    seconds = time.perf_counter() - start
    size_bytes = 0
    for trajectory in trajectories:
        for state in trajectory:
            for value in state.values():
                size_bytes += value.numel() * value.element_size()
    return trajectories, seconds, size_bytes / (1024.0 * 1024.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D MTT-style trajectory matching baseline for MedMNIST3D")
    parser.add_argument("--dataset", default="organmnist3d")
    parser.add_argument("--ipc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-experts", type=int, default=1)
    parser.add_argument("--expert-epochs", type=int, default=3)
    parser.add_argument("--expert-steps-per-epoch", type=int, default=10)
    parser.add_argument("--expert-batch-per-class", type=int, default=4)
    parser.add_argument("--expert-lr", type=float, default=0.01)
    parser.add_argument("--expert-momentum", type=float, default=0.0)
    parser.add_argument("--expert-weight-decay", type=float, default=0.0)
    parser.add_argument("--mtt-iterations", type=int, default=20)
    parser.add_argument("--syn-steps", type=int, default=3)
    parser.add_argument("--expert-window", type=int, default=1)
    parser.add_argument("--syn-lr", type=float, default=0.01)
    parser.add_argument("--lr-syn", type=float, default=0.1)
    parser.add_argument("--init-noise-std", type=float, default=0.01)
    parser.add_argument("--mtt-model", choices=["convnet3d", "resnet18_3d", "resnet10_3d_gn", "resnet18_3d_gn"], default="convnet3d")
    parser.add_argument("--mtt-width", type=int, default=16)
    parser.add_argument("--eval-model", choices=["convnet3d", "resnet18_3d", "resnet10_3d_gn", "resnet18_3d_gn"], default="convnet3d")
    parser.add_argument("--eval-width", type=int, default=32)
    parser.add_argument("--eval-epochs", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-optimizer", choices=["adamw", "sgd"], default="sgd")
    parser.add_argument("--eval-lr", type=float, default=1e-3)
    parser.add_argument("--eval-momentum", type=float, default=0.9)
    parser.add_argument("--eval-weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-class-weight-mode", choices=["none", "inverse_freq"], default="none")
    parser.add_argument("--eval-balanced-sampler", action="store_true")
    parser.add_argument("--select-metric", choices=["auc_acc", "auc", "acc", "balanced_acc", "auc_bacc"], default="auc_bacc")
    parser.add_argument("--method-name", default="3d_mtt_official_style")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "medmnist")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_start = time.perf_counter()

    trajectories, buffer_seconds, buffer_size_mb = train_expert_trajectories(
        args, x_train, y_train, in_channels, num_classes, device
    )

    x_syn_init, y_syn = init_synthetic_from_real(
        x_train.cpu(), y_train.cpu(), num_classes, args.ipc, args.seed, args.init_noise_std
    )
    x_syn = nn.Parameter(x_syn_init.to(device))
    y_syn = y_syn.to(device)
    optimizer_syn = torch.optim.SGD([x_syn], lr=args.lr_syn, momentum=0.5)
    distill_start = time.perf_counter()

    template_model = build_model(args.mtt_model, in_channels=in_channels, num_classes=num_classes, width=args.mtt_width).to(device)
    template_model.eval()
    template_params = {k: v.detach().clone() for k, v in template_model.named_parameters()}
    buffers = {k: v.detach().clone() for k, v in template_model.named_buffers()}
    param_names = list(template_params.keys())

    for it in range(1, args.mtt_iterations + 1):
        trajectory = random.choice(trajectories)
        max_start = max(0, len(trajectory) - args.expert_window - 1)
        start_idx = random.randint(0, max_start)
        start_params = {k: v.to(device) for k, v in trajectory[start_idx].items()}
        target_params = {k: v.to(device) for k, v in trajectory[start_idx + args.expert_window].items()}
        student_flat = flatten_named(start_params).detach().requires_grad_(True)
        target_flat = flatten_named(target_params).detach()
        start_flat = flatten_named(start_params).detach()

        for _ in range(args.syn_steps):
            student_params = unflatten_named(student_flat, {name: start_params[name] for name in param_names})
            call_state = {**student_params, **buffers}
            logits = functional_call(template_model, call_state, (x_syn,))
            ce = F.cross_entropy(logits, y_syn)
            grad = torch.autograd.grad(ce, student_flat, create_graph=True)[0]
            student_flat = student_flat - args.syn_lr * grad

        param_loss = F.mse_loss(student_flat, target_flat, reduction="sum")
        param_dist = F.mse_loss(start_flat, target_flat, reduction="sum").clamp_min(1e-12)
        loss = param_loss / param_dist
        optimizer_syn.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_syn.step()
        with torch.no_grad():
            x_syn.clamp_(0.0, 1.0)
        if it == 1 or it % args.log_every == 0 or it == args.mtt_iterations:
            print(f"[{args.dataset}] mtt_iter={it:04d}/{args.mtt_iterations} loss={float(loss.item()):.8e}")

    distill_seconds = time.perf_counter() - distill_start
    peak_memory_mb = float("nan")
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    distill_runtime_sec = buffer_seconds + distill_seconds
    distill_peak_memory_mb = round(peak_memory_mb, 3) if not np.isnan(peak_memory_mb) else peak_memory_mb

    val_loader = make_loader(x_val, y_val, args.eval_batch_size, False, args.num_workers)
    test_loader = make_loader(x_test, y_test, args.eval_batch_size, False, args.num_workers)
    eval_start = time.perf_counter()
    eval_metrics = eval_synthetic(
        x_syn.detach(), y_syn.detach(), val_loader, test_loader, args, task, num_classes, in_channels
    )
    eval_seconds = time.perf_counter() - eval_start
    total_seconds = time.perf_counter() - total_start

    run_name = f"{args.dataset}_{args.method_name}_ipc{args.ipc}_{args.mtt_model}_w{args.mtt_width}_seed{args.seed}"
    run_dir = args.output_dir / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    syn_path = run_dir / "synthetic.pt"
    torch.save({"images": x_syn.detach().cpu(), "labels": y_syn.detach().cpu(), "args": vars(args)}, syn_path)
    row = {
        "dataset": args.dataset,
        "method": args.method_name,
        "ipc": args.ipc,
        "seed": args.seed,
        "dd_method": "mtt",
        "mtt_iterations": args.mtt_iterations,
        "syn_steps": args.syn_steps,
        "expert_window": args.expert_window,
        "num_experts": args.num_experts,
        "expert_epochs": args.expert_epochs,
        "expert_steps_per_epoch": args.expert_steps_per_epoch,
        "mtt_model": args.mtt_model,
        "mtt_width": args.mtt_width,
        "eval_model": args.eval_model,
        "eval_width": args.eval_width,
        "eval_epochs": args.eval_epochs,
        "train_size": int(y_syn.numel()),
        "full_train_size": len(train_ds),
        "synthetic_path": str(syn_path),
        "distilled_object": "synthetic_3d_volume",
        "requires_expert_buffer": True,
        "requires_unrolled_inner_training": True,
        "buffer_seconds": round(buffer_seconds, 3),
        "buffer_size_mb": round(buffer_size_mb, 3),
        "distill_seconds": round(distill_seconds, 3),
        "distill_runtime_sec": round(distill_runtime_sec, 3),
        "distill_seconds_per_iteration": round(distill_seconds / max(1, args.mtt_iterations), 6),
        "distill_runtime_sec_per_iteration": round(distill_runtime_sec / max(1, args.mtt_iterations), 6),
        "eval_seconds": round(eval_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "peak_memory_mb": distill_peak_memory_mb,
        "distill_peak_memory_mb": distill_peak_memory_mb,
        "status": "ok",
        **{k: round(v, 6) if isinstance(v, float) else v for k, v in eval_metrics.items()},
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    append_csv(args.output_dir / "metrics" / "medmnist3d_2d_to3d_mtt_runs.csv", row)
    print(f"TEST {json.dumps(row, indent=2)}")


if __name__ == "__main__":
    main()
