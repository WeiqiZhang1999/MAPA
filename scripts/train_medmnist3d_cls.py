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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models3d import build_model  # noqa: E402


DATASETS_3D = [
    "organmnist3d",
    "nodulemnist3d",
    "adrenalmnist3d",
    "fracturemnist3d",
    "vesselmnist3d",
    "synapsemnist3d",
    "mosmed_binary96",
    "mosmed_binary128",
]

CUSTOM_3D_INFO = {
    "mosmed_binary96": {
        "task": "binary-class",
        "label": {"0": "CT-0", "1": "CT-1/2/3/4"},
        "python_class": "LocalNPZVolumeDataset",
        "npz_name": "mosmed_binary96.npz",
    },
    "mosmed_binary128": {
        "task": "binary-class",
        "label": {"0": "CT-0", "1": "CT-1/2/3/4"},
        "python_class": "LocalNPZVolumeDataset",
        "npz_name": "mosmed_binary128.npz",
    },
}


def local_npz_path(name: str, root: Path) -> Path:
    return root / f"{name}.npz"


def local_npz_summary_path(name: str, root: Path | None = None) -> Path:
    if root is None:
        root = ROOT / "data" / "medmnist"
    return root / f"{name}_summary.json"


def load_local_npz_info(name: str, root: Path | None = None) -> dict | None:
    if root is None:
        root = ROOT / "data" / "medmnist"
    npz_path = local_npz_path(name, root)
    if not npz_path.exists():
        return None
    summary_path = local_npz_summary_path(name, root)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        labels = summary.get("label") or summary.get("labels")
        if labels is None:
            class_counts = summary.get("class_counts", {})
            labels = {str(k): str(k) for k in sorted(class_counts.keys(), key=lambda x: int(x))}
        return {
            "task": summary.get("task", "binary-class"),
            "label": {str(k): str(v) for k, v in labels.items()},
            "python_class": "LocalNPZVolumeDataset",
            "npz_name": f"{name}.npz",
        }
    data = np.load(npz_path)
    labels_arr = np.concatenate([data[f"{split}_labels"].reshape(-1) for split in ("train", "val", "test")])
    classes = sorted(np.unique(labels_arr).astype(int).tolist())
    return {
        "task": "binary-class" if len(classes) == 2 else "multi-class",
        "label": {str(c): str(c) for c in classes},
        "python_class": "LocalNPZVolumeDataset",
        "npz_name": f"{name}.npz",
    }


class LocalNPZVolumeDataset:
    def __init__(self, imgs: np.ndarray, labels: np.ndarray) -> None:
        self.imgs = imgs
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_medmnist_dataset(name: str, split: str, root: Path):
    import medmnist
    from medmnist import INFO

    local_info = CUSTOM_3D_INFO.get(name) or load_local_npz_info(name, root)
    if local_info is not None:
        npz_path = root / local_info["npz_name"]
        if not npz_path.exists():
            raise FileNotFoundError(
                f"{npz_path} not found. Prepare the local NPZ dataset before running experiments."
            )
        data = np.load(npz_path)
        return LocalNPZVolumeDataset(data[f"{split}_images"], data[f"{split}_labels"])
    if name not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset: {name}")
    dataset_cls = getattr(medmnist, INFO[name]["python_class"])
    return dataset_cls(split=split, root=str(root), download=True, as_rgb=False)


def get_dataset_info(name: str, root: Path | None = None) -> dict:
    if name in CUSTOM_3D_INFO:
        return CUSTOM_3D_INFO[name]
    local_info = load_local_npz_info(name, root)
    if local_info is not None:
        return local_info
    from medmnist import INFO

    return INFO[name]


def to_volume_tensor(images: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(images).float()
    if x.ndim == 4:
        x = x.unsqueeze(1)
    elif x.ndim == 5 and x.shape[-1] == 1:
        x = x.permute(0, 4, 1, 2, 3)
    else:
        raise ValueError(f"Unexpected image array shape: {tuple(x.shape)}")
    return x / 255.0


def to_label_tensor(labels: np.ndarray) -> torch.Tensor:
    y = torch.from_numpy(labels)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y[:, 0]
    return y.long()


def prepare_tensors(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    return to_volume_tensor(dataset.imgs), to_label_tensor(dataset.labels)


def select_random_coreset(x: torch.Tensor, y: torch.Tensor, num_classes: int, ipc: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for c in range(num_classes):
        class_indices = torch.nonzero(y == c, as_tuple=False).flatten()
        if class_indices.numel() < ipc:
            raise ValueError(f"Class {c} has only {class_indices.numel()} samples, cannot select ipc={ipc}")
        perm = torch.randperm(class_indices.numel(), generator=generator)[:ipc]
        selected.append(class_indices[perm])
    indices = torch.cat(selected, dim=0)
    return x[indices].clone(), y[indices].clone(), indices.tolist()


def select_index_coreset(x: torch.Tensor, y: torch.Tensor, indices_path: Path) -> tuple[torch.Tensor, torch.Tensor, list[int], dict]:
    with indices_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    indices = [int(i) for i in payload["indices"]]
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return x[index_tensor].clone(), y[index_tensor].clone(), indices, payload


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    ds = torch.utils.data.TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)


def infer_task(info: dict) -> str:
    return info.get("task", "multi-class")


def compute_metrics(logits: np.ndarray, labels: np.ndarray, task: str, num_classes: int) -> dict[str, float]:
    if task == "multi-label, binary-class":
        probs = 1.0 / (1.0 + np.exp(-logits))
        pred = (probs > 0.5).astype(np.int64)
        acc = float((pred == labels).mean())
        auc = float(roc_auc_score(labels, probs, average="macro"))
        return {"auc": auc, "acc": acc}

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = probs.argmax(axis=1)
    acc = float(accuracy_score(labels, pred))
    balanced_acc = float(balanced_accuracy_score(labels, pred))
    per_class_recall = recall_score(labels, pred, labels=list(range(num_classes)), average=None, zero_division=0)
    if num_classes == 2:
        auc = float(roc_auc_score(labels, probs[:, 1]))
    else:
        auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    metrics = {"auc": auc, "acc": acc, "balanced_acc": balanced_acc}
    for c, recall in enumerate(per_class_recall):
        metrics[f"recall_c{c}"] = float(recall)
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device, limit_batches: int | None = None, progress: bool = True) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for step, (x, y) in enumerate(tqdm(loader, desc="train", leave=False, disable=not progress)):
        if limit_batches is not None and step >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, task: str, num_classes: int, limit_batches: int | None = None, progress: bool = True) -> dict[str, float]:
    model.eval()
    logits_all = []
    labels_all = []
    total_loss = 0.0
    total = 0
    for step, (x, y) in enumerate(tqdm(loader, desc="eval", leave=False, disable=not progress)):
        if limit_batches is not None and step >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(y.detach().cpu().numpy())

    logits_np = np.concatenate(logits_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    metrics = compute_metrics(logits_np, labels_np, task=task, num_classes=num_classes)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def is_better(current: dict[str, float], best: dict[str, float], select_metric: str) -> bool:
    eps = 1e-12
    if select_metric == "auc":
        return current["auc"] > best["auc"] + eps
    if select_metric == "acc":
        return current["acc"] > best["acc"] + eps
    if select_metric == "balanced_acc":
        return current.get("balanced_acc", -1.0) > best.get("balanced_acc", -1.0) + eps
    if select_metric == "auc_bacc":
        if current["auc"] > best["auc"] + eps:
            return True
        return abs(current["auc"] - best["auc"]) <= eps and current.get("balanced_acc", -1.0) > best.get("balanced_acc", -1.0) + eps
    if current["auc"] > best["auc"] + eps:
        return True
    if abs(current["auc"] - best["auc"]) <= eps and current["acc"] > best["acc"] + eps:
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedMNIST3D full-data classification baseline")
    parser.add_argument("--dataset", default="organmnist3d")
    parser.add_argument("--model", choices=["convnet3d", "resnet18_3d", "resnet10_3d_gn", "resnet18_3d_gn"], default="convnet3d")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "medmnist")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--select-metric", choices=["auc_acc", "auc", "acc", "balanced_acc", "auc_bacc"], default="auc_acc")
    parser.add_argument("--train-subset", choices=["full", "random_coreset", "index_coreset"], default="full")
    parser.add_argument("--ipc", type=int, default=None)
    parser.add_argument("--indices-file", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    return parser.parse_args()


def run_dataset(args: argparse.Namespace, dataset_name: str) -> dict:
    seed_everything(args.seed)
    started = time.perf_counter()
    info = get_dataset_info(dataset_name, args.data_root)
    task = infer_task(info)
    num_classes = len(info["label"])
    train_ds = load_medmnist_dataset(dataset_name, "train", args.data_root)
    val_ds = load_medmnist_dataset(dataset_name, "val", args.data_root)
    test_ds = load_medmnist_dataset(dataset_name, "test", args.data_root)

    x_train, y_train = prepare_tensors(train_ds)
    x_val, y_val = prepare_tensors(val_ds)
    x_test, y_test = prepare_tensors(test_ds)
    original_train_size = len(train_ds)
    selected_indices = None
    coreset_method = args.train_subset
    coreset_space = ""
    if args.train_subset == "random_coreset":
        if args.ipc is None:
            raise ValueError("--ipc is required when --train-subset random_coreset")
        x_train, y_train, selected_indices = select_random_coreset(x_train, y_train, num_classes, args.ipc, args.seed)
        coreset_method = "volume_random"
        coreset_space = "voxel"
    elif args.train_subset == "index_coreset":
        if args.indices_file is None:
            raise ValueError("--indices-file is required when --train-subset index_coreset")
        x_train, y_train, selected_indices, selection_payload = select_index_coreset(x_train, y_train, args.indices_file)
        coreset_method = selection_payload.get("method", "index_coreset")
        coreset_space = selection_payload.get("space", "")

    train_loader = make_loader(x_train, y_train, args.batch_size, True, args.num_workers)
    val_loader = make_loader(x_val, y_val, args.batch_size, False, args.num_workers)
    test_loader = make_loader(x_test, y_test, args.batch_size, False, args.num_workers)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(args.model, in_channels=x_train.shape[1], num_classes=num_classes, width=args.width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.train_subset == "full":
        subset_tag = "full"
    elif args.train_subset == "index_coreset":
        subset_tag = f"index_coreset_{coreset_method}_ipc{args.ipc}"
    else:
        subset_tag = f"{args.train_subset}_ipc{args.ipc}"
    run_name = f"{dataset_name}_{subset_tag}_{args.model}_w{args.width}_seed{args.seed}_{args.select_metric}"
    ckpt_dir = args.output_dir / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    best_val_metrics = {"auc": -1.0, "acc": -1.0, "balanced_acc": -1.0, "loss": float("inf")}
    best_epoch = -1
    history_path = ckpt_dir / "history.csv"
    if history_path.exists():
        history_path.unlink()

    train_runtime_sec = 0.0
    val_runtime_sec = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_train_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.limit_train_batches, progress=not args.no_progress)
        train_runtime_sec += time.perf_counter() - epoch_train_start
        scheduler.step()
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            print(f"[{dataset_name}] epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f}")
            append_csv(
                history_path,
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_auc": "",
                    "val_acc": "",
                    "val_loss": "",
                },
            )
            continue

        val_start = time.perf_counter()
        val_metrics = evaluate(model, val_loader, criterion, device, task, num_classes, args.limit_eval_batches, progress=not args.no_progress)
        val_runtime_sec += time.perf_counter() - val_start
        print(
            f"[{dataset_name}] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['acc']:.4f}"
        )
        append_csv(
            history_path,
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_auc": round(val_metrics["auc"], 6),
                "val_acc": round(val_metrics["acc"], 6),
                "val_loss": round(val_metrics["loss"], 6),
            },
        )
        if is_better(val_metrics, best_val_metrics, args.select_metric):
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "dataset": dataset_name,
                    "selected_indices": selected_indices,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_start = time.perf_counter()
    test_metrics = evaluate(model, test_loader, criterion, device, task, num_classes, args.limit_eval_batches, progress=not args.no_progress)
    test_runtime_sec = time.perf_counter() - test_start
    runtime_sec = time.perf_counter() - started
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    row = {
        "dataset": dataset_name,
        "model": args.model,
        "width": args.width,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_subset": args.train_subset,
        "coreset_method": coreset_method,
        "coreset_space": coreset_space,
        "indices_file": str(args.indices_file) if args.indices_file is not None else "",
        "ipc": args.ipc if args.ipc is not None else "",
        "eval_every": args.eval_every,
        "select_metric": args.select_metric,
        "best_epoch": best_epoch,
        "val_auc": round(best_val_metrics["auc"], 6),
        "val_acc": round(best_val_metrics["acc"], 6),
        "test_auc": round(test_metrics["auc"], 6),
        "test_acc": round(test_metrics["acc"], 6),
        "test_balanced_acc": round(test_metrics.get("balanced_acc", float("nan")), 6),
        "test_loss": round(test_metrics["loss"], 6),
        "train_size": int(x_train.shape[0]),
        "full_train_size": original_train_size,
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "runtime_sec": round(runtime_sec, 3),
        "train_runtime_sec": round(train_runtime_sec, 3),
        "val_runtime_sec": round(val_runtime_sec, 3),
        "test_runtime_sec": round(test_runtime_sec, 3),
        "train_seconds_per_epoch": round(train_runtime_sec / max(1, args.epochs), 6),
        "peak_memory_mb": round(peak_memory_mb, 3),
    }
    metrics_path = args.output_dir / "metrics" / "medmnist3d_cls_runs.csv"
    append_csv(metrics_path, row)
    with (ckpt_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    print(f"TEST {json.dumps(row, indent=2)}")
    print(f"metrics_csv={metrics_path}")
    return row


def main() -> None:
    args = parse_args()
    datasets = DATASETS_3D if args.dataset == "all" else [args.dataset]
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device}")
    print(f"torch={torch.__version__}")
    for dataset in datasets:
        run_dataset(args, dataset)


if __name__ == "__main__":
    main()
