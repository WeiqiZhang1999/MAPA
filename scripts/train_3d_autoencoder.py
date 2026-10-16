from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image, ImageDraw
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_medmnist3d_cls import (  # noqa: E402
    append_csv,
    load_medmnist_dataset,
    make_loader,
    prepare_tensors,
    seed_everything,
)


def group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DownBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, block: str = "plain") -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )
        self.refine = make_refine_block(out_ch, block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, block: str = "plain") -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )
        self.refine = make_refine_block(out_ch, block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.up(x))


class PlainRefineBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualRefineBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ConvNeXtRefineBlock3D(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
            nn.GroupNorm(group_count(channels), channels),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def make_refine_block(channels: int, block: str) -> nn.Module:
    if block == "plain":
        return PlainRefineBlock3D(channels)
    if block == "residual":
        return ResidualRefineBlock3D(channels)
    if block == "convnext":
        return ConvNeXtRefineBlock3D(channels)
    raise ValueError(f"Unknown AE block: {block}")


class ConvAutoencoder3D(nn.Module):
    """Lightweight spatial-latent 3D autoencoder for latent-level DD pilots."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        latent_channels: int = 64,
        block: str = "plain",
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels must contain at least two stages")
        enc = []
        prev = in_channels
        for ch in channels:
            enc.append(DownBlock3D(prev, ch, block=block))
            prev = ch
        self.encoder = nn.Sequential(*enc)
        self.to_latent = nn.Conv3d(prev, latent_channels, kernel_size=1)
        self.from_latent = nn.Conv3d(latent_channels, prev, kernel_size=1)
        dec = []
        for ch in reversed(channels[:-1]):
            dec.append(UpBlock3D(prev, ch, block=block))
            prev = ch
        dec.append(nn.ConvTranspose3d(prev, in_channels, kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x))

    def decode(self, z: torch.Tensor, output_shape: tuple[int, int, int] | None = None) -> torch.Tensor:
        x = torch.sigmoid(self.decoder(self.from_latent(z)))
        if output_shape is not None and tuple(x.shape[-3:]) != tuple(output_shape):
            x = F.interpolate(x, size=output_shape, mode="trilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z, output_shape=tuple(x.shape[-3:])), z


def parse_channels(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("--channels must contain at least one integer")
    return values


def gradient_loss_3d(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = torch.zeros((), device=recon.device)
    for dim in (2, 3, 4):
        recon_diff = recon.diff(dim=dim)
        target_diff = target.diff(dim=dim)
        loss = loss + F.l1_loss(recon_diff, target_diff)
    return loss / 3.0


def ssim_loss_3d(recon: torch.Tensor, target: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool3d(recon, kernel_size=kernel_size, stride=1, padding=pad)
    mu_y = F.avg_pool3d(target, kernel_size=kernel_size, stride=1, padding=pad)
    sigma_x = F.avg_pool3d(recon * recon, kernel_size=kernel_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool3d(target * target, kernel_size=kernel_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool3d(recon * target, kernel_size=kernel_size, stride=1, padding=pad) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return 1.0 - ssim.mean()


def reconstruction_metrics(recon: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    recon = recon.detach().float().cpu()
    target = target.detach().float().cpu()
    mse = F.mse_loss(recon, target).item()
    l1 = F.l1_loss(recon, target).item()
    psnr = float(10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item())
    return {"recon_l1": float(l1), "recon_mse": float(mse), "recon_psnr": psnr}


def volume_slice_uint8(volume: torch.Tensor) -> Image.Image:
    vol = volume.detach().float().cpu()
    if vol.ndim == 4:
        vol = vol[0]
    center = vol.shape[0] // 2
    img = vol[center].clamp(0.0, 1.0).numpy()
    return Image.fromarray((img * 255.0).round().astype("uint8"), mode="L").convert("RGB")


def save_reconstruction_grid(x: torch.Tensor, recon: torch.Tensor, path: Path, max_items: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().cpu()
    recon = recon.detach().cpu()
    n = min(max_items, x.shape[0])
    tile_w = int(x.shape[-1])
    tile_h = int(x.shape[-2])
    scale = max(1, min(4, 112 // max(tile_w, 1)))
    disp_w = tile_w * scale
    disp_h = tile_h * scale
    label_h = 18
    canvas = Image.new("RGB", (3 * disp_w, n * (disp_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["input", "recon", "abs diff"]
    for row in range(n):
        diff = (x[row] - recon[row]).abs()
        tiles = [volume_slice_uint8(x[row]), volume_slice_uint8(recon[row]), volume_slice_uint8(diff)]
        for col, tile in enumerate(tiles):
            x0 = col * disp_w
            y0 = row * (disp_h + label_h)
            draw.text((x0 + 2, y0 + 2), labels[col], fill=(0, 0, 0))
            canvas.paste(tile.resize((disp_w, disp_h), Image.Resampling.NEAREST), (x0, y0 + label_h))
    canvas.save(path)


@torch.no_grad()
def evaluate_reconstruction(model: nn.Module, loader, device: torch.device, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    total = 0
    for step, (x, _) in enumerate(tqdm(loader, desc="ae-eval", leave=False)):
        if max_batches is not None and step >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        recon, _ = model(x)
        total_l1 += F.l1_loss(recon, x, reduction="sum").item()
        total_mse += F.mse_loss(recon, x, reduction="sum").item()
        total += x.numel()
    mse = total_mse / max(total, 1)
    l1 = total_l1 / max(total, 1)
    psnr = float(10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item())
    return {"recon_l1": float(l1), "recon_mse": float(mse), "recon_psnr": psnr}


def train_autoencoder(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_ds = load_medmnist_dataset(args.dataset, "train", args.data_root)
    val_ds = load_medmnist_dataset(args.dataset, "val", args.data_root)
    x_train, y_train = prepare_tensors(train_ds)
    x_val, y_val = prepare_tensors(val_ds)
    train_loader = make_loader(x_train, y_train, args.batch_size, True, args.num_workers)
    val_loader = make_loader(x_val, y_val, args.batch_size, False, args.num_workers)

    model = ConvAutoencoder3D(
        in_channels=x_train.shape[1],
        channels=parse_channels(args.channels),
        latent_channels=args.latent_channels,
        block=args.block,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    best_state = None
    best_val = {"recon_l1": float("inf")}
    best_epoch = -1
    start = time.perf_counter()
    peak_memory_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for step, (x, _) in enumerate(tqdm(train_loader, desc=f"ae-train {epoch}", leave=False, disable=args.no_progress)):
            if args.limit_train_batches is not None and step >= args.limit_train_batches:
                break
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon, _ = model(x)
            loss = F.l1_loss(recon, x) + args.mse_weight * F.mse_loss(recon, x)
            if args.grad_weight > 0:
                loss = loss + args.grad_weight * gradient_loss_3d(recon, x)
            if args.ssim_weight > 0:
                loss = loss + args.ssim_weight * ssim_loss_3d(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
            total += x.size(0)
        scheduler.step()
        val_metrics = evaluate_reconstruction(model, val_loader, device, args.limit_eval_batches)
        train_loss = total_loss / max(total, 1)
        print(
            f"[ae] epoch={epoch} train_loss={train_loss:.6f} "
            f"val_l1={val_metrics['recon_l1']:.6f} val_mse={val_metrics['recon_mse']:.6f}"
        )
        if val_metrics["recon_l1"] < best_val["recon_l1"]:
            best_val = val_metrics
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    runtime = time.perf_counter() - start
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        sample_train_x = x_train[: min(args.num_examples, x_train.shape[0])].to(device)
        sample_train_recon, sample_train_z = model(sample_train_x)
        sample_val_x = x_val[: min(args.num_examples, x_val.shape[0])].to(device)
        sample_val_recon, sample_val_z = model(sample_val_x)
    run_dir = args.output_dir / "autoencoder" / f"{args.dataset}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "dataset": args.dataset,
            "seed": args.seed,
            "channels": parse_channels(args.channels),
            "latent_channels": args.latent_channels,
            "block": args.block,
            "input_shape": tuple(x_train.shape[1:]),
            "latent_shape": tuple(sample_val_z.shape[1:]),
            "best_epoch": best_epoch,
            "best_val": best_val,
            "args": serializable_args,
        },
        ckpt_path,
    )
    torch.save(
        {
            "x": sample_val_x.detach().cpu(),
            "recon": sample_val_recon.detach().cpu(),
            "z": sample_val_z.detach().cpu(),
        },
        run_dir / "recon_examples.pt",
    )
    torch.save(
        {
            "x": sample_train_x.detach().cpu(),
            "recon": sample_train_recon.detach().cpu(),
            "z": sample_train_z.detach().cpu(),
        },
        run_dir / "train_recon_examples.pt",
    )
    torch.save(
        {
            "x": sample_val_x.detach().cpu(),
            "recon": sample_val_recon.detach().cpu(),
            "z": sample_val_z.detach().cpu(),
        },
        run_dir / "val_recon_examples.pt",
    )
    recon_png = run_dir / "val_recon_central_slices.png"
    train_recon_png = run_dir / "train_recon_central_slices.png"
    save_reconstruction_grid(sample_val_x, sample_val_recon, recon_png, args.num_examples)
    save_reconstruction_grid(sample_train_x, sample_train_recon, train_recon_png, args.num_examples)
    row = {
        "dataset": args.dataset,
        "seed": args.seed,
        "method": "3d_autoencoder",
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path),
        "recon_examples": str(run_dir / "recon_examples.pt"),
        "recon_png": str(recon_png),
        "train_recon_examples": str(run_dir / "train_recon_examples.pt"),
        "train_recon_png": str(train_recon_png),
        "val_recon_examples": str(run_dir / "val_recon_examples.pt"),
        "val_recon_png": str(recon_png),
        "runtime_sec": runtime,
        "peak_memory_mb": peak_memory_mb,
        **best_val,
    }
    append_csv(args.output_dir / "metrics" / "autoencoder_runs.csv", row)
    (run_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight 3D autoencoder for LatentDD-style pilots")
    parser.add_argument("--dataset", default="mosmed_binary96")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", default="16,32,64,128")
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--block", choices=["plain", "residual", "convnext"], default="plain")
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--grad-weight", type=float, default=0.0)
    parser.add_argument("--ssim-weight", type=float, default=0.0)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "medmnist")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "latent3d_dd_pilot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = train_autoencoder(args)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
