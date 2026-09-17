#!/usr/bin/env python
"""Train Full SER-CMT under the established CMT/DALFE RetinaNet protocol."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator

import train_m2cai as benchmark
from ser_cmt import (
    STAGE_DIMS,
    SERCMTFeatures,
    architecture_manifest,
    build_ser_cmt,
    deform_conv2d_flop_jit,
    ecsa_modules,
)


RESULT_FIELDS = (
    "epoch",
    "train_loss",
    "loss_classification",
    "loss_bbox_regression",
    "val_loss",
    "val_loss_classification",
    "val_loss_bbox_regression",
    "precision",
    "recall",
    "mAP50",
    "mAP50_95",
    "lr",
    "epoch_seconds",
    "val_images",
)

FAIRNESS_FIELDS = (
    "image_size",
    "batch_size",
    "eval_batch_size",
    "lr",
    "weight_decay",
    "eta_min",
    "amp",
)

PAPER_SPLITS = {"train": 1405, "val": 843, "test": 563}


def build_detector(image_size: int, use_valid_mask: bool = True) -> Tuple[nn.Module, nn.Module]:
    base = build_ser_cmt(
        img_size=image_size,
        drop_path_rate=0.1,
        use_valid_mask=use_valid_mask,
    )
    for name in ("_fc", "_bn", "_swish", "_avg_pooling", "_drop", "pre_logits", "head"):
        setattr(base, name, nn.Identity())
    body = SERCMTFeatures(base, infer_valid_mask=use_valid_mask)
    backbone = benchmark.BackboneFPN(body, STAGE_DIMS, out_channels=128)
    bases = (16, 32, 64, 128, 256)
    sizes = tuple(tuple(int(round(x * (2 ** (k / 3)))) for k in range(3)) for x in bases)
    ratios = tuple((0.5, 1.0, 2.0) for _ in bases)
    anchors = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
    detector = RetinaNet(
        backbone=backbone,
        num_classes=len(benchmark.CLASS_NAMES) + 1,
        anchor_generator=anchors,
        min_size=image_size,
        max_size=image_size,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        score_thresh=0.01,
        nms_thresh=0.5,
        detections_per_img=300,
        topk_candidates=1000,
    )
    detector._benchmark_flop_handles = {
        "torchvision::deform_conv2d": deform_conv2d_flop_jit,
    }
    detector._benchmark_variant = "SER-CMT-Ti (DALFE@Stage1,2 + DALFE/ECSA@all Stage3 blocks)"
    return detector, base


def append_csv(path: Path, row: Dict[str, object]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
        stream.flush()
        os.fsync(stream.fileno())


def append_result_files(
    canonical_path: Path,
    requested_results_path: Path,
    row: Dict[str, object],
    warning_path: Path,
) -> bool:
    """Persist the canonical journal even when Excel/WPS locks results.csv.

    Windows spreadsheet applications commonly hold an exclusive handle on an
    opened CSV.  The append-only ``metrics.csv`` journal is therefore written
    first and is authoritative.  ``results.csv`` remains a live mirror when it
    is writable; a mirror failure is recorded but must never terminate a long
    GPU run.
    """
    append_csv(canonical_path, row)
    try:
        append_csv(requested_results_path, row)
        return True
    except PermissionError as exc:
        append_jsonl(
            warning_path,
            {
                "epoch": row.get("epoch"),
                "file": str(requested_results_path),
                "error": f"{type(exc).__name__}: {exc}",
                "fallback": str(canonical_path),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        print(
            f"WARNING: results.csv is locked; epoch {row.get('epoch')} is safe in "
            f"{canonical_path.name}",
            flush=True,
        )
        return False


def synchronize_results(canonical_path: Path, requested_results_path: Path) -> bool:
    """Best-effort final synchronization of the requested results.csv name."""
    try:
        shutil.copyfile(canonical_path, requested_results_path)
        return True
    except PermissionError:
        return False


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_best_row(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return max(rows, key=lambda row: float(row["mAP50_95"])) if rows else {}


def load_dalfe_initialization(model: nn.Module, checkpoint_path: Optional[Path]) -> dict:
    if checkpoint_path is None:
        return {
            "source": None,
            "initialization": "random CMT-Ti plus module initializers",
            "loaded_keys": 0,
            "parameter_load_rate": 0.0,
            "note": "Pass --dalfe-checkpoint to reproduce the manuscript initialization.",
        }
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DALFE checkpoint not found: {checkpoint_path}")
    checkpoint = benchmark.torch_load_path(checkpoint_path, map_location="cpu")
    source = checkpoint["model"] if "model" in checkpoint else checkpoint
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    result = model.load_state_dict(compatible, strict=False)
    loaded_numel = sum(target[key].numel() for key in compatible)
    total_numel = sum(value.numel() for value in target.values())
    trainable_total = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "source": str(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "source_best_mAP50_95": checkpoint.get("best_mAP50_95"),
        "loaded_keys": len(compatible),
        "target_keys": len(target),
        "key_load_rate": len(compatible) / max(len(target), 1),
        "loaded_numel": loaded_numel,
        "target_state_numel": total_numel,
        "parameter_load_rate": loaded_numel / max(total_numel, 1),
        "trainable_parameters": trainable_total,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "shape_mismatches": [
            {
                "key": key,
                "source": list(value.shape),
                "target": list(target[key].shape),
            }
            for key, value in source.items()
            if key in target and tuple(value.shape) != tuple(target[key].shape)
        ],
    }
    return report


def enforce_fairness(args, reference_config: Optional[dict], dataset_root: Path) -> dict:
    current = {
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "eta_min": args.eta_min,
        "amp": args.amp,
    }
    mismatches = {}
    if reference_config:
        mismatches = {
            key: {"reference": reference_config.get(key), "SER-CMT": current[key]}
            for key in FAIRNESS_FIELDS
            if reference_config.get(key) != current[key]
        }
        if reference_config.get("dataset"):
            reference_dataset = Path(reference_config["dataset"]).resolve()
            if reference_dataset != dataset_root.resolve():
                mismatches["dataset"] = {
                    "reference": str(reference_dataset),
                    "SER-CMT": str(dataset_root.resolve()),
                }
        if mismatches:
            raise ValueError(f"Fairness configuration mismatch: {mismatches}")
    return {
        "reference": "user-supplied config" if reference_config else "published protocol defaults",
        "matched_fields": current,
        "dataset": str(dataset_root.resolve()),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "gradient_clip_norm": 5.0,
        "detector": "torchvision RetinaNet + 128-channel FPN",
        "augmentation": "VocToolDataset deterministic letterbox + identical train-only color jitter/flip",
        "warmup": "none (identical to Baseline/DALFE)",
        "loss": "unchanged torchvision RetinaNet classification and box regression losses",
        "nms_and_postprocessing": "unchanged score_thresh=0.01, nms_thresh=0.5, detections_per_img=300, topk=1000",
    }


@torch.no_grad()
def validation_loss(model, loader, device: torch.device, amp: bool) -> dict:
    """Compute deterministic validation loss without updating BN or consuming RNG."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    training_states = [(module, module.training) for module in model.modules()]
    totals: Dict[str, float] = {}
    batches = 0
    try:
        model.eval()
        # RetinaNet's root flag selects the loss-return path; all children stay
        # in eval mode, so BN/dropout/drop-path cannot alter model state.
        model.training = True
        for images, targets in loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation loss: {loss_dict}")
            totals["val_loss"] = totals.get("val_loss", 0.0) + float(loss)
            for key, value in loss_dict.items():
                totals[f"val_loss_{key}"] = totals.get(f"val_loss_{key}", 0.0) + float(value)
            batches += 1
    finally:
        for module, state in training_states:
            module.training = state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return {key: value / max(batches, 1) for key, value in totals.items()}


def edge_diagnostics(base: nn.Module, epoch: int) -> dict:
    alpha, beta = base.semantic_edge_prior.fusion_weights()
    modules = []
    for name, module in ecsa_modules(base):
        structural = module.last_structural_attention
        modules.append(
            {
                "module": name,
                "eta": float(module.eta.detach().cpu()),
                "gamma": float(module.gamma.detach().cpu()),
                "mec_mean": float(structural.mean()) if structural is not None else None,
                "mec_std": float(structural.std()) if structural is not None else None,
                "mec_min": float(structural.min()) if structural is not None else None,
                "mec_max": float(structural.max()) if structural is not None else None,
            }
        )
    maps = base.semantic_edge_prior.last_maps
    return {
        "epoch": int(epoch),
        "alpha": alpha,
        "beta": beta,
        "eta": [item["eta"] for item in modules],
        "gamma": [item["gamma"] for item in modules],
        "edge2_mean": float(maps["edge2"].mean()) if maps else None,
        "edge2_std": float(maps["edge2"].std()) if maps else None,
        "modules": modules,
    }


def snapshot_sources(model_root: Path, run_dir: Path) -> None:
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in (
        "ser_cmt.py",
        "train_ser_cmt.py",
        "train_m2cai.py",
        "cmt.py",
        "README.md",
    ):
        source = model_root / name
        if source.exists():
            shutil.copy2(source, snapshot / name)


def run_smoke(args, model, train_loader, val_loader, device, base, load_report):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    train_values = benchmark.train_one_epoch(
        model, train_loader, optimizer, scaler, device, args.amp, max_batches=1
    )
    val_values = benchmark.evaluate(model, val_loader, device, args.amp, max_batches=1)
    diagnostics = edge_diagnostics(base, epoch=0)
    print(
        json.dumps(
            {
                "smoke": "ok",
                "train": train_values,
                "val": val_values,
                "weight_loading": load_report,
                "edge_diagnostics": diagnostics,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_training(args) -> None:
    model_root = Path(__file__).resolve().parent
    dataset_root = (
        Path(args.dataset)
        if args.dataset
        else model_root.parent / "datasets" / "m2cai16-tool-locations"
    )
    dalfe_checkpoint = Path(args.dalfe_checkpoint) if args.dalfe_checkpoint else None
    reference_config = None
    if args.reference_config:
        reference_config_path = Path(args.reference_config)
        reference_config = json.loads(reference_config_path.read_text(encoding="utf-8"))
    fairness = enforce_fairness(args, reference_config, dataset_root)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is required; --allow-cpu is diagnostics-only")
    benchmark.seed_everything(args.seed)
    print(f"Building Full SER-CMT on {device} from {model_root}", flush=True)
    model, base = build_detector(args.image_size, use_valid_mask=args.use_valid_mask)
    load_report = load_dalfe_initialization(model, dalfe_checkpoint)
    model.to(device)

    train_set = benchmark.VocToolDataset(dataset_root, "train", args.image_size, augment=True)
    val_set = benchmark.VocToolDataset(dataset_root, "val", args.image_size, augment=False)
    test_set = benchmark.VocToolDataset(dataset_root, "test", args.image_size, augment=False)
    expected_splits = reference_config.get("splits", PAPER_SPLITS) if reference_config else PAPER_SPLITS
    actual_splits = {"train": len(train_set), "val": len(val_set), "test": len(test_set)}
    if actual_splits != expected_splits:
        raise ValueError(f"Dataset split mismatch: expected {expected_splits}, got {actual_splits}")
    if args.smoke:
        train_set = Subset(train_set, list(range(min(args.batch_size, len(train_set)))))
        val_set = Subset(val_set, [0])
    train_loader = benchmark.make_loader(train_set, args.batch_size, args.workers, True, args.seed)
    val_loader = benchmark.make_loader(val_set, args.eval_batch_size, args.workers, False, args.seed + 1)
    if args.smoke:
        run_smoke(args, model, train_loader, val_loader, device, base, load_report)
        return

    run_dir = (
        Path(args.output_dir)
        if args.output_dir
        else model_root / "runs" / f"ser_cmt_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.csv"
    compatibility_metrics_path = run_dir / "metrics.csv"
    diagnostics_path = run_dir / "ecsa_diagnostics.jsonl"
    manifest = architecture_manifest(base)
    config = {
        "model": "SER-CMT",
        "full_name": "Surgical Edge-Robust CMT",
        "seed": args.seed,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "workers": args.workers,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "eta_min": args.eta_min,
        "warmup": "none",
        "amp": args.amp,
        "gradient_clip_norm": 5.0,
        "detector": "torchvision RetinaNet + 128-channel FPN",
        "dataset": str(dataset_root),
        "splits": actual_splits,
        "classes": list(benchmark.CLASS_NAMES),
        "pretrained_initialization": str(dalfe_checkpoint) if dalfe_checkpoint else None,
        "use_valid_mask": args.use_valid_mask,
        "architecture": manifest,
        "fairness": fairness,
        "torch": torch.__version__,
        "torchvision": importlib.import_module("torchvision").__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "targets": {
            "DALFE_mAP50": 0.8403076861569228,
            "DALFE_mAP50_95": 0.42891674085128384,
        },
    }
    benchmark.atomic_json(run_dir / "config.json", config)
    benchmark.atomic_json(run_dir / "model_config.json", manifest)
    benchmark.atomic_json(run_dir / "train_args.json", vars(args))
    benchmark.atomic_json(run_dir / "weight_loading_report.json", load_report)
    snapshot_sources(model_root, run_dir)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    start_epoch, best_map = 1, -1.0
    last_path = run_dir / "last.pt"
    if args.resume and last_path.exists():
        checkpoint = benchmark.torch_load_path(last_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map = float(checkpoint.get("best_mAP50_95", -1.0))
        print(f"Resuming at epoch {start_epoch}", flush=True)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for path in (results_path, compatibility_metrics_path, diagnostics_path):
            if path.exists():
                path.replace(path.with_name(f"{path.stem}_incomplete_{timestamp}{path.suffix}"))

    status = {
        "state": "RUNNING",
        "model": "SER-CMT",
        "seed": args.seed,
        "epoch": start_epoch - 1,
        "epochs": args.epochs,
        "initialized_from": str(dalfe_checkpoint) if dalfe_checkpoint else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    benchmark.atomic_json(run_dir / "status.json", status)
    if start_epoch == 1:
        append_jsonl(diagnostics_path, edge_diagnostics(base, epoch=0))

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_values = benchmark.train_one_epoch(
                model, train_loader, optimizer, scaler, device, args.amp
            )
            val_losses = validation_loss(model, val_loader, device, args.amp)
            val_values = benchmark.evaluate(model, val_loader, device, args.amp)
            elapsed = time.perf_counter() - epoch_start
            row = {
                "epoch": epoch,
                "train_loss": train_values.get("train_loss", 0.0),
                "loss_classification": train_values.get("loss_classification", 0.0),
                "loss_bbox_regression": train_values.get("loss_bbox_regression", 0.0),
                "val_loss": val_losses.get("val_loss", 0.0),
                "val_loss_classification": val_losses.get("val_loss_classification", 0.0),
                "val_loss_bbox_regression": val_losses.get("val_loss_bbox_regression", 0.0),
                "precision": val_values["precision"],
                "recall": val_values["recall"],
                "mAP50": val_values["mAP50"],
                "mAP50_95": val_values["mAP50_95"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": elapsed,
                "val_images": val_values["images"],
            }
            results_mirror_written = append_result_files(
                compatibility_metrics_path,
                results_path,
                row,
                run_dir / "io_warnings.jsonl",
            )
            append_jsonl(run_dir / "val.log", {"epoch": epoch, **val_losses, **val_values})
            scheduler.step()
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_mAP50_95": max(best_map, val_values["mAP50_95"]),
                "config": config,
            }
            benchmark.atomic_torch_save(last_path, checkpoint)
            if val_values["mAP50_95"] > best_map:
                best_map = val_values["mAP50_95"]
                checkpoint["best_mAP50_95"] = best_map
                benchmark.atomic_torch_save(run_dir / "best.pt", checkpoint)
                benchmark.atomic_json(
                    run_dir / "best_val_metrics.json", {"epoch": epoch, **val_losses, **val_values}
                )
            if epoch % 10 == 0:
                append_jsonl(diagnostics_path, edge_diagnostics(base, epoch=epoch))
            status.update(
                {
                    "epoch": epoch,
                    "last_metrics": row,
                    "best_mAP50_95": best_map,
                    "results_csv_mirror_current": results_mirror_written,
                    "canonical_metrics_file": str(compatibility_metrics_path),
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            benchmark.atomic_json(run_dir / "status.json", status)
            print(json.dumps({"model": "SER-CMT", "seed": args.seed, **row}), flush=True)

        checkpoint = benchmark.torch_load_path(run_dir / "best.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        test_loader = benchmark.make_loader(
            test_set, args.eval_batch_size, args.workers, False, args.seed + 2
        )
        test_values = benchmark.evaluate(model, test_loader, device, args.amp)
        benchmark.atomic_json(run_dir / "test_metrics.json", test_values)
        profile = benchmark.profile_detector(model, device, args.image_size, args.seed)
        benchmark.atomic_json(run_dir / "profile.json", profile)
        best_row = read_best_row(compatibility_metrics_path)
        results_synchronized = synchronize_results(compatibility_metrics_path, results_path)
        summary = {
            "model": "SER-CMT",
            "seed": args.seed,
            "best_validation": best_row,
            "test": test_values,
            "profile": profile,
            "beats_DALFE_mAP50": float(best_row["mAP50"]) > 0.8403076861569228,
            "beats_DALFE_mAP50_95": float(best_row["mAP50_95"]) > 0.42891674085128384,
            "results_csv_synchronized": results_synchronized,
            "canonical_metrics_file": str(compatibility_metrics_path),
        }
        benchmark.atomic_json(run_dir / "summary.json", summary)
        status.update(
            {
                "state": "COMPLETED",
                "test": test_values,
                "profile": profile,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        benchmark.atomic_json(run_dir / "status.json", status)
    except Exception as exc:
        status.update(
            {
                "state": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        benchmark.atomic_json(run_dir / "status.json", status)
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="")
    parser.add_argument("--dalfe-checkpoint", default="")
    parser.add_argument("--reference-config", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-valid-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
