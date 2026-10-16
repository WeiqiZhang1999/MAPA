#!/usr/bin/env python
"""Summarize autoencoder cost and latent/decoded storage accounting.

This is a post-hoc accounting utility: it reads existing autoencoder
``result.json`` / ``best.pt`` files and latent-DD ``result.json`` /
``synthetic_decoded.pt`` files without rerunning experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


def bytes_to_mib(value: int | float | None) -> float | str:
    if value is None:
        return ""
    return round(float(value) / (1024.0 * 1024.0), 6)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_result_jsons(roots: list[Path], require_key: str | None = None) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "result.json":
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(root.rglob("result.json"))
        else:
            continue
        for candidate in candidates:
            if require_key is None:
                paths.append(candidate)
                continue
            try:
                row = load_json(candidate)
            except Exception:
                continue
            if require_key in row:
                paths.append(candidate)
    return sorted(set(paths))


def tensor_bytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    return 0


def tensor_shape(obj: Any) -> str:
    if torch.is_tensor(obj):
        return "x".join(str(v) for v in obj.shape)
    return ""


def state_group(name: str) -> str:
    if name.startswith("encoder.") or name.startswith("to_latent."):
        return "encoder"
    if name.startswith("from_latent.") or name.startswith("decoder."):
        return "decoder"
    return "other"


def summarize_autoencoder_result(result_path: Path, repo_root: Path) -> dict[str, Any]:
    row = load_json(result_path)
    ckpt = Path(row.get("checkpoint", ""))
    if not ckpt.is_absolute():
        ckpt = repo_root / ckpt

    out: dict[str, Any] = {
        "source_result": str(result_path),
        "dataset": row.get("dataset", ""),
        "seed": row.get("seed", ""),
        "best_epoch": row.get("best_epoch", ""),
        "runtime_sec": row.get("runtime_sec", ""),
        "peak_memory_mb": row.get("peak_memory_mb", ""),
        "recon_l1": row.get("recon_l1", ""),
        "recon_mse": row.get("recon_mse", ""),
        "recon_psnr": row.get("recon_psnr", ""),
        "checkpoint": str(ckpt),
        "checkpoint_bytes": ckpt.stat().st_size if ckpt.exists() else "",
        "checkpoint_mib": bytes_to_mib(ckpt.stat().st_size) if ckpt.exists() else "",
        "status": "ok" if ckpt.exists() else "missing_checkpoint",
    }
    if not ckpt.exists():
        return out

    payload = torch.load(ckpt, map_location="cpu")
    state = payload.get("model_state", {})
    groups = {
        "encoder": {"params": 0, "bytes": 0},
        "decoder": {"params": 0, "bytes": 0},
        "other": {"params": 0, "bytes": 0},
    }
    total_params = 0
    total_bytes = 0
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        n_params = int(value.numel())
        n_bytes = int(value.numel() * value.element_size())
        group = state_group(name)
        groups[group]["params"] += n_params
        groups[group]["bytes"] += n_bytes
        total_params += n_params
        total_bytes += n_bytes

    out.update(
        {
            "input_shape": "x".join(str(v) for v in payload.get("input_shape", [])),
            "latent_shape": "x".join(str(v) for v in payload.get("latent_shape", [])),
            "total_params": total_params,
            "total_param_bytes": total_bytes,
            "total_param_mib": bytes_to_mib(total_bytes),
            "encoder_params": groups["encoder"]["params"],
            "encoder_param_bytes": groups["encoder"]["bytes"],
            "encoder_param_mib": bytes_to_mib(groups["encoder"]["bytes"]),
            "decoder_params": groups["decoder"]["params"],
            "decoder_param_bytes": groups["decoder"]["bytes"],
            "decoder_param_mib": bytes_to_mib(groups["decoder"]["bytes"]),
            "other_params": groups["other"]["params"],
            "other_param_bytes": groups["other"]["bytes"],
            "other_param_mib": bytes_to_mib(groups["other"]["bytes"]),
        }
    )
    return out


def summarize_synthetic_result(result_path: Path, repo_root: Path) -> dict[str, Any]:
    row = load_json(result_path)
    synthetic_path = Path(row.get("synthetic_path", ""))
    if not synthetic_path.is_absolute():
        synthetic_path = repo_root / synthetic_path
    ae_path = Path(row.get("autoencoder_checkpoint", ""))
    if not ae_path.is_absolute():
        ae_path = repo_root / ae_path

    out: dict[str, Any] = {
        "source_result": str(result_path),
        "dataset": row.get("dataset", ""),
        "method": row.get("method", ""),
        "ipc": row.get("ipc", ""),
        "seed": row.get("seed", ""),
        "lpc_multiplier": row.get("lpc_multiplier", ""),
        "decoded_ipc": row.get("decoded_ipc", ""),
        "num_synthetic_volumes": row.get("num_synthetic_volumes", ""),
        "latent_storage_ratio_reported": row.get("latent_storage_ratio", ""),
        "storage_budget_ipc": row.get("storage_budget_ipc", ""),
        "synthetic_path": str(synthetic_path),
        "synthetic_file_bytes": synthetic_path.stat().st_size if synthetic_path.exists() else "",
        "synthetic_file_mib": bytes_to_mib(synthetic_path.stat().st_size) if synthetic_path.exists() else "",
        "autoencoder_checkpoint": str(ae_path),
        "ae_checkpoint_bytes": ae_path.stat().st_size if ae_path.exists() else "",
        "ae_checkpoint_mib": bytes_to_mib(ae_path.stat().st_size) if ae_path.exists() else "",
        "status": "ok" if synthetic_path.exists() else "missing_synthetic",
    }
    if not synthetic_path.exists():
        return out

    payload = torch.load(synthetic_path, map_location="cpu")
    x_bytes = tensor_bytes(payload.get("x_syn"))
    z_final_bytes = tensor_bytes(payload.get("z_final"))
    z_syn_bytes = tensor_bytes(payload.get("z_syn"))
    z_init_bytes = tensor_bytes(payload.get("z_init"))
    y_bytes = tensor_bytes(payload.get("y_syn"))
    mode_bytes = tensor_bytes(payload.get("mode_syn"))
    latent_bytes = z_final_bytes or z_syn_bytes

    decoder_param_bytes = ""
    decoder_param_mib: float | str = ""
    latent_plus_decoder_bytes: int | str = ""
    latent_plus_decoder_mib: float | str = ""
    if ae_path.exists():
        ae_payload = torch.load(ae_path, map_location="cpu")
        state = ae_payload.get("model_state", {})
        decoder_param_bytes_int = 0
        for name, value in state.items():
            if torch.is_tensor(value) and state_group(name) == "decoder":
                decoder_param_bytes_int += int(value.numel() * value.element_size())
        decoder_param_bytes = decoder_param_bytes_int
        decoder_param_mib = bytes_to_mib(decoder_param_bytes_int)
        latent_plus_decoder_bytes = latent_bytes + decoder_param_bytes_int
        latent_plus_decoder_mib = bytes_to_mib(latent_plus_decoder_bytes)

    out.update(
        {
            "x_syn_shape": tensor_shape(payload.get("x_syn")),
            "z_final_shape": tensor_shape(payload.get("z_final")),
            "decoded_volume_tensor_bytes": x_bytes,
            "decoded_volume_tensor_mib": bytes_to_mib(x_bytes),
            "latent_code_tensor_bytes": latent_bytes,
            "latent_code_tensor_mib": bytes_to_mib(latent_bytes),
            "z_syn_tensor_bytes": z_syn_bytes,
            "z_init_tensor_bytes": z_init_bytes,
            "label_tensor_bytes": y_bytes,
            "mode_tensor_bytes": mode_bytes,
            "tensor_latent_storage_ratio": round(x_bytes / latent_bytes, 6) if latent_bytes else "",
            "latent_plus_decoder_bytes": latent_plus_decoder_bytes,
            "latent_plus_decoder_mib": latent_plus_decoder_mib,
            "decoder_param_bytes": decoder_param_bytes,
            "decoder_param_mib": decoder_param_mib,
            "latent_plus_full_ae_checkpoint_bytes": latent_bytes + (ae_path.stat().st_size if ae_path.exists() else 0),
            "latent_plus_full_ae_checkpoint_mib": bytes_to_mib(
                latent_bytes + (ae_path.stat().st_size if ae_path.exists() else 0)
            ),
        }
    )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ae-root", type=Path, action="append", default=[])
    parser.add_argument("--ae-result", type=Path, action="append", default=[])
    parser.add_argument("--synthetic-root", type=Path, action="append", default=[])
    parser.add_argument("--synthetic-result", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ae_paths = discover_result_jsons(args.ae_root, require_key="checkpoint") + args.ae_result
    synthetic_paths = discover_result_jsons(args.synthetic_root, require_key="synthetic_path") + args.synthetic_result
    ae_rows = [summarize_autoencoder_result(path.resolve(), repo_root) for path in sorted(set(ae_paths))]
    storage_rows = [summarize_synthetic_result(path.resolve(), repo_root) for path in sorted(set(synthetic_paths))]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ae_csv = args.output_dir / "ae_cost_accounting.csv"
    storage_csv = args.output_dir / "storage_accounting.csv"
    write_csv(ae_csv, ae_rows)
    write_csv(storage_csv, storage_rows)
    summary = {
        "ae_rows": len(ae_rows),
        "storage_rows": len(storage_rows),
        "ae_csv": str(ae_csv),
        "storage_csv": str(storage_csv),
    }
    (args.output_dir / "accounting_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
