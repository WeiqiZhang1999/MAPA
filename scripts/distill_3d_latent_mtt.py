import argparse
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

from distill_3d_dm import append_dm_csv, eval_synthetic  # noqa: E402
from distill_3d_latent_dd import (  # noqa: E402
    encode_volume_bank,
    init_latents,
    latent_storage_ratio,
    load_autoencoder,
    save_synthetic_grid,
)
from distill_3d_mtt import flatten_named, sample_class_balanced, unflatten_named  # noqa: E402
from models3d import build_model  # noqa: E402
from train_medmnist3d_cls import (  # noqa: E402
    get_dataset_info,
    infer_task,
    load_medmnist_dataset,
    make_loader,
    prepare_tensors,
    seed_everything,
)


def snapshot_model_state(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "params": {k: v.detach().cpu().clone() for k, v in model.named_parameters()},
        "buffers": {k: v.detach().cpu().clone() for k, v in model.named_buffers()},
    }


def train_latent_expert_trajectories(
    args: argparse.Namespace,
    z_real: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> tuple[list[list[dict[str, dict[str, torch.Tensor]]]], float, float]:
    start = time.perf_counter()
    trajectories: list[list[dict[str, dict[str, torch.Tensor]]]] = []
    for expert_id in range(args.num_experts):
        model = build_model(
            args.mtt_model,
            in_channels=z_real.shape[1],
            num_classes=num_classes,
            width=args.mtt_width,
        ).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.expert_lr,
            momentum=args.expert_momentum,
            weight_decay=args.expert_weight_decay,
        )
        trajectory = [snapshot_model_state(model)]
        for epoch in range(1, args.expert_epochs + 1):
            model.train()
            for _ in range(max(1, args.expert_steps_per_epoch)):
                zb, yb = sample_class_balanced(
                    z_real,
                    y_train,
                    num_classes,
                    args.expert_batch_per_class,
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(zb), yb)
                loss.backward()
                optimizer.step()
            trajectory.append(snapshot_model_state(model))
            print(f"[latent expert {expert_id}] epoch={epoch}/{args.expert_epochs} loss={float(loss.item()):.6f}")
        trajectories.append(trajectory)
    seconds = time.perf_counter() - start
    size_bytes = 0
    for trajectory in trajectories:
        for state in trajectory:
            for group in state.values():
                for value in group.values():
                    size_bytes += value.numel() * value.element_size()
    return trajectories, seconds, size_bytes / (1024.0 * 1024.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official MTT-style trajectory matching on 3D autoencoder latents")
    parser.add_argument("--dataset", default="mosmed_binary96")
    parser.add_argument("--ipc", type=int, default=5)
    parser.add_argument("--lpc-multiplier", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-init", choices=["kmeans", "random", "mixed"], default="random")
    parser.add_argument("--mixed-random-frac", type=float, default=0.5)
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--expert-epochs", type=int, default=5)
    parser.add_argument("--expert-steps-per-epoch", type=int, default=20)
    parser.add_argument("--expert-batch-per-class", type=int, default=4)
    parser.add_argument("--expert-lr", type=float, default=0.01)
    parser.add_argument("--expert-momentum", type=float, default=0.0)
    parser.add_argument("--expert-weight-decay", type=float, default=0.0)
    parser.add_argument("--mtt-iterations", type=int, default=200)
    parser.add_argument("--syn-steps", type=int, default=5)
    parser.add_argument("--expert-window", type=int, default=2)
    parser.add_argument("--syn-lr", type=float, default=0.01)
    parser.add_argument("--lr-syn", type=float, default=0.005)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--mtt-model", choices=["convnet3d", "resnet18_3d", "resnet10_3d_gn", "resnet18_3d_gn"], default="convnet3d")
    parser.add_argument("--mtt-width", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--method-name", default="Latent-MTT")
    parser.add_argument("--eval-model", choices=["convnet3d", "resnet18_3d", "resnet10_3d_gn", "resnet18_3d_gn"], default="convnet3d")
    parser.add_argument("--eval-width", type=int, default=32)
    parser.add_argument("--eval-epochs", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-seed", type=int, default=None)
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
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_start = time.perf_counter()

    info = get_dataset_info(args.dataset, args.data_root)
    task = infer_task(info)
    num_classes = len(info["label"])
    train_ds = load_medmnist_dataset(args.dataset, "train", args.data_root)
    val_ds = load_medmnist_dataset(args.dataset, "val", args.data_root)
    test_ds = load_medmnist_dataset(args.dataset, "test", args.data_root)
    x_train, y_train_cpu = prepare_tensors(train_ds)
    x_val, y_val = prepare_tensors(val_ds)
    x_test, y_test = prepare_tensors(test_ds)
    val_loader = make_loader(x_val, y_val, args.eval_batch_size, False, args.num_workers)
    test_loader = make_loader(x_test, y_test, args.eval_batch_size, False, args.num_workers)

    ae, ae_payload = load_autoencoder(args.autoencoder_checkpoint, device)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    storage_ratio = latent_storage_ratio(ae_payload)
    z_real_cpu = encode_volume_bank(ae, x_train, args.encode_batch_size, device)
    z_init_cpu, y_syn_cpu = init_latents(
        z_real_cpu,
        y_train_cpu,
        num_classes,
        args.ipc,
        args.lpc_multiplier,
        args.seed,
        args.latent_init,
        args.mixed_random_frac,
    )
    decoded_ipc = args.ipc * args.lpc_multiplier
    storage_budget_ipc = decoded_ipc / storage_ratio if storage_ratio == storage_ratio else float("nan")

    z_real = z_real_cpu.to(device)
    y_train = y_train_cpu.to(device)
    trajectories, buffer_seconds, buffer_size_mb = train_latent_expert_trajectories(
        args, z_real, y_train, num_classes, device
    )

    z_syn = nn.Parameter(z_init_cpu.to(device))
    y_syn = y_syn_cpu.to(device)
    optimizer_syn = torch.optim.SGD([z_syn], lr=args.lr_syn, momentum=0.5)
    distill_start = time.perf_counter()

    template_model = build_model(
        args.mtt_model,
        in_channels=z_syn.shape[1],
        num_classes=num_classes,
        width=args.mtt_width,
    ).to(device)
    template_model.eval()
    template_params = {k: v.detach().clone() for k, v in template_model.named_parameters()}
    param_names = list(template_params.keys())
    best_loss = float("inf")
    best_z = z_syn.detach().cpu().clone()

    for it in range(1, args.mtt_iterations + 1):
        trajectory = random.choice(trajectories)
        max_start = max(0, len(trajectory) - args.expert_window - 1)
        start_idx = random.randint(0, max_start)
        start_state = trajectory[start_idx]
        target_state = trajectory[start_idx + args.expert_window]
        start_params = {k: v.to(device) for k, v in start_state["params"].items()}
        target_params = {k: v.to(device) for k, v in target_state["params"].items()}
        start_buffers = {k: v.to(device) for k, v in start_state["buffers"].items()}
        student_flat = flatten_named(start_params).detach().requires_grad_(True)
        target_flat = flatten_named(target_params).detach()
        start_flat = flatten_named(start_params).detach()

        for _ in range(args.syn_steps):
            student_params = unflatten_named(student_flat, {name: start_params[name] for name in param_names})
            call_state = {**student_params, **start_buffers}
            logits = functional_call(template_model, call_state, (z_syn,))
            ce = F.cross_entropy(logits, y_syn)
            grad = torch.autograd.grad(ce, student_flat, create_graph=True)[0]
            student_flat = student_flat - args.syn_lr * grad

        param_loss = F.mse_loss(student_flat, target_flat, reduction="sum")
        param_dist = F.mse_loss(start_flat, target_flat, reduction="sum").clamp_min(1e-12)
        trajectory_loss = param_loss / param_dist
        anchor = F.mse_loss(z_syn, z_init_cpu.to(device))
        loss = trajectory_loss + args.anchor_weight * anchor
        optimizer_syn.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_syn.step()
        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_z = z_syn.detach().cpu().clone()
        if it == 1 or it % args.log_every == 0 or it == args.mtt_iterations:
            print(f"[latent-mtt] iter={it:04d}/{args.mtt_iterations} loss={float(loss.item()):.8e} traj={float(trajectory_loss.item()):.8e} anchor={float(anchor.item()):.8e}")

    distill_runtime = buffer_seconds + (time.perf_counter() - distill_start)
    distill_peak_memory_mb = 0.0
    if device.type == "cuda":
        distill_peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        torch.cuda.reset_peak_memory_stats(device)

    decode_export_start = time.perf_counter()
    with torch.no_grad():
        z_final = best_z.to(device)
        flat_final = z_final.flatten(1)
        flat_init = z_init_cpu.to(device).flatten(1)
        latent_delta_l2 = (flat_final - flat_init).norm(dim=1)
        init_l2 = flat_init.norm(dim=1).clamp_min(1e-8)
        latent_delta_rel = latent_delta_l2 / init_l2
        x_syn = ae.decode(z_final, output_shape=tuple(x_train.shape[-3:])).detach().cpu().clamp(0.0, 1.0)
    decode_export_runtime = time.perf_counter() - decode_export_start
    decode_export_peak_memory_mb = 0.0
    if device.type == "cuda":
        decode_export_peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        torch.cuda.reset_peak_memory_stats(device)

    eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    eval_metrics = {}
    if not args.skip_eval:
        seed_everything(eval_seed)
        eval_metrics = eval_synthetic(
            x_syn=x_syn,
            y_syn=y_syn_cpu,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            task=task,
            num_classes=num_classes,
            in_channels=x_train.shape[1],
        )
    eval_peak_memory_mb = 0.0
    if device.type == "cuda" and not args.skip_eval:
        eval_peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    method_slug = args.method_name.lower().replace(" ", "_").replace("/", "_")
    run_name = f"{args.dataset}_{method_slug}_mtt_{args.latent_init}_ipc{args.ipc}_lpcx{args.lpc_multiplier}_seed{args.seed}"
    run_dir = args.output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = run_dir / "synthetic_decoded.pt"
    synthetic_png = run_dir / "synthetic_central_slices.png"
    save_synthetic_grid(x_syn, y_syn_cpu, synthetic_png)
    torch.save(
        {
            "x_syn": x_syn,
            "y_syn": y_syn_cpu,
            "z_syn": best_z,
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
        },
        synthetic_path,
    )

    total_runtime = time.perf_counter() - total_start
    train_class_counts = [int((y_train_cpu == c).sum().item()) for c in range(num_classes)]
    row = {
        "dataset": args.dataset,
        "method": args.method_name,
        "distilled_object": "decoded_3d_volume_from_latent",
        "latent_metric": "mtt",
        "latent_init": args.latent_init,
        "ipc": args.ipc,
        "lpc_multiplier": args.lpc_multiplier,
        "decoded_ipc": decoded_ipc,
        "num_synthetic_volumes": int(y_syn_cpu.numel()),
        "latent_storage_ratio": storage_ratio,
        "storage_budget_ipc": storage_budget_ipc,
        "seed": args.seed,
        "eval_seed": eval_seed,
        "train_class_counts": json.dumps(train_class_counts),
        "status": "completed",
        "error_type": "",
        "eval_skipped": bool(args.skip_eval),
        "test_balanced_acc": eval_metrics.get("test_balanced_acc", float("nan")),
        "test_auc": eval_metrics.get("test_auc", float("nan")),
        "test_positive_recall": eval_metrics.get("test_recall_c1", float("nan")),
        "test_negative_recall": eval_metrics.get("test_recall_c0", float("nan")),
        "eval_best_epoch": eval_metrics.get("eval_best_epoch", -1),
        "distill_runtime_sec": distill_runtime,
        "decode_export_runtime_sec": decode_export_runtime,
        "runtime_sec": total_runtime,
        "peak_memory_mb": distill_peak_memory_mb,
        "distill_peak_memory_mb": distill_peak_memory_mb,
        "decode_export_peak_memory_mb": decode_export_peak_memory_mb,
        "eval_peak_memory_mb": eval_peak_memory_mb,
        "best_latent_loss": best_loss,
        "anchor_weight": args.anchor_weight,
        "mtt_iterations": args.mtt_iterations,
        "syn_steps": args.syn_steps,
        "expert_window": args.expert_window,
        "num_experts": args.num_experts,
        "expert_epochs": args.expert_epochs,
        "expert_steps_per_epoch": args.expert_steps_per_epoch,
        "expert_buffer_runtime_sec": buffer_seconds,
        "expert_buffer_size_mb": buffer_size_mb,
        "mtt_model": args.mtt_model,
        "mtt_width": args.mtt_width,
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
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
