#!/usr/bin/env python
"""Unified RetinaNet-FPN benchmark for M2CAI16 tool detection.

The file is copied into each model directory.  It deliberately keeps the
detector, input pipeline, optimizer, evaluation and profiling identical while
only changing the backbone implementation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import os
import random
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


MODEL_NAMES = ("CMT", "Conformer", "DefMamba", "FastViT", "UniFormer")
CLASS_NAMES = ("Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag")
METRIC_FIELDS = (
    "epoch", "train_loss", "loss_classification", "loss_bbox_regression",
    "precision", "recall", "mAP50", "mAP50_95", "lr", "epoch_seconds", "val_images",
)


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def atomic_torch_save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # PyTorch 2.0's Windows zip writer cannot open some non-ASCII paths by
    # filename.  A Python-owned binary stream keeps checkpoint I/O Unicode-safe.
    with tmp.open("wb") as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def torch_load_path(path: Path, map_location="cpu"):
    with path.open("rb") as stream:
        return torch.load(stream, map_location=map_location)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class VocToolDataset(Dataset):
    """VOC annotations with deterministic square letterboxing."""

    def __init__(self, root: Path, split: str, image_size: int = 320, augment: bool = False):
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.augment = bool(augment)
        split_path = self.root / "ImageSets" / "Main" / f"{split}.txt"
        self.ids = [x.strip() for x in split_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.class_to_id = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
        self.annotations = [self._read_annotation(image_id) for image_id in self.ids]

    def _read_annotation(self, image_id: str) -> Tuple[np.ndarray, np.ndarray]:
        root = ET.parse(self.root / "Annotations" / f"{image_id}.xml").getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in self.class_to_id:
                continue
            node = obj.find("bndbox")
            if node is None:
                continue
            x1 = float(node.findtext("xmin", "0"))
            y1 = float(node.findtext("ymin", "0"))
            x2 = float(node.findtext("xmax", "0"))
            y2 = float(node.findtext("ymax", "0"))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
                labels.append(self.class_to_id[name])
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ids)

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        for enhancer, strength in (
            (ImageEnhance.Brightness, 0.15),
            (ImageEnhance.Contrast, 0.15),
            (ImageEnhance.Color, 0.10),
        ):
            image = enhancer(image).enhance(1.0 + random.uniform(-strength, strength))
        return image

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        image = Image.open(self.root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
        boxes = self.annotations[index][0].copy()
        labels = self.annotations[index][1].copy()
        width, height = image.size
        scale = min(self.image_size / width, self.image_size / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        pad_x = (self.image_size - new_w) // 2
        pad_y = (self.image_size - new_h) // 2
        canvas = Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
        canvas.paste(image, (pad_x, pad_y))
        image = canvas
        if len(boxes):
            boxes[:, (0, 2)] = boxes[:, (0, 2)] * scale + pad_x
            boxes[:, (1, 3)] = boxes[:, (1, 3)] * scale + pad_y
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, self.image_size)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, self.image_size)
        if self.augment:
            image = self._color_jitter(image)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if len(boxes):
                    old_x1 = boxes[:, 0].copy()
                    boxes[:, 0] = self.image_size - boxes[:, 2]
                    boxes[:, 2] = self.image_size - old_x1
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(array.copy())
        box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.as_tensor(labels, dtype=torch.int64)
        area = ((box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1])) if len(box_tensor) else torch.zeros(0)
        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(label_tensor),), dtype=torch.int64),
        }
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


class CMTFeatures(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def _stage(self, x: Tensor, patch: nn.Module, blocks: Iterable[nn.Module], rel: Tensor):
        batch = x.shape[0]
        tokens, (height, width) = patch(x)
        for block in blocks:
            tokens = block(tokens, height, width, rel)
        feature = tokens.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x: Tensor) -> List[Tensor]:
        m = self.model
        x = m.stem_norm1(m.stem_relu1(m.stem_conv1(x)))
        x = m.stem_norm2(m.stem_relu2(m.stem_conv2(x)))
        x = m.stem_norm3(m.stem_relu3(m.stem_conv3(x)))
        a = self._stage(x, m.patch_embed_a, m.blocks_a, m.relative_pos_a)
        b = self._stage(a, m.patch_embed_b, m.blocks_b, m.relative_pos_b)
        c = self._stage(b, m.patch_embed_c, m.blocks_c, m.relative_pos_c)
        d = self._stage(c, m.patch_embed_d, m.blocks_d, m.relative_pos_d)
        return [a, b, c, d]


class ConformerFeatures(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> List[Tensor]:
        m = self.model
        batch = x.shape[0]
        cls_tokens = m.cls_token.expand(batch, -1, -1)
        x_base = m.maxpool(m.act1(m.bn1(m.conv1(x))))
        x = m.conv_1(x_base, return_x_2=False)
        x_t = m.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = m.trans_1(torch.cat([cls_tokens, x_t], dim=1))
        stage1 = stage2 = stage3 = None
        for i in range(2, m.fin_stage):
            x, x_t = getattr(m, f"conv_trans_{i}")(x, x_t)
            if i == 4:
                stage1 = x
            elif i == 8:
                stage2 = x
            elif i == 12:
                stage3 = x
        assert stage1 is not None and stage2 is not None and stage3 is not None
        stage4 = F.max_pool2d(stage3, kernel_size=2, stride=2)
        return [stage1, stage2, stage3, stage4]


class UniFormerFeatures(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @staticmethod
    def _blocks(x: Tensor, blocks: Iterable[nn.Module]) -> Tensor:
        for block in blocks:
            x = block(x)
        return x

    def forward(self, x: Tensor) -> List[Tensor]:
        m = self.model
        a = self._blocks(m.pos_drop(m.patch_embed1(x)), m.blocks1)
        b = self._blocks(m.patch_embed2(a), m.blocks2)
        c = self._blocks(m.patch_embed3(b), m.blocks3)
        d = self._blocks(m.patch_embed4(c), m.blocks4)
        return [a, b, c, d]


class ListFeatures(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> List[Tensor]:
        return list(self.model(x))


class BackboneFPN(nn.Module):
    def __init__(self, body: nn.Module, channels: Sequence[int], out_channels: int = 128):
        super().__init__()
        self.body = body
        self.fpn = FeaturePyramidNetwork(list(channels), out_channels, extra_blocks=LastLevelMaxPool())
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> OrderedDict:
        features = self.body(x)
        ordered = OrderedDict((str(i), feature) for i, feature in enumerate(features))
        return self.fpn(ordered)


def build_feature_body(model_name: str, model_root: Path, image_size: int):
    sys.path.insert(0, str(model_root))
    flop_handles = {}
    if model_name == "CMT":
        module = importlib.import_module("cmt")
        base = module.cmt_ti(pretrained=False, img_size=image_size, num_classes=0, drop_path_rate=0.1)
        for name in ("_fc", "_bn", "_swish", "_avg_pooling", "_drop", "pre_logits", "head"):
            setattr(base, name, nn.Identity())
        return CMTFeatures(base), [46, 92, 184, 368], "CMT-Ti", flop_handles
    if model_name == "Conformer":
        module = importlib.import_module("conformer")
        base = module.Conformer(
            patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
            num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=0, drop_path_rate=0.1,
        )
        base.trans_norm = nn.Identity()
        base.trans_cls_head = nn.Identity()
        base.pooling = nn.Identity()
        base.conv_cls_head = nn.Identity()
        return ConformerFeatures(base), [64, 128, 256, 256], "Conformer-Tiny", flop_handles
    if model_name == "DefMamba":
        module = importlib.import_module("classification.models.vmamba")
        base = module.Backbone_VSSM(
            depths=[2, 2, 5, 2], dims=48, ssm_d_state=16, ssm_dt_rank="auto",
            ssm_ratio=1.0, mlp_ratio=4.0, downsample_version="v3",
            patchembed_version="v2", drop_path_rate=0.2, pretrained=None,
        )
        flop_handles["prim::PythonOp.SelectiveScan"] = module.selective_scan_flop_jit
        return ListFeatures(base), [48, 96, 192, 384], "DefMamba-Tiny", flop_handles
    if model_name == "FastViT":
        module = importlib.import_module("models.fastvit")
        base = module.fastvit_t8(pretrained=False, fork_feat=True)
        return ListFeatures(base), [48, 96, 192, 384], "FastViT-T8", flop_handles
    if model_name == "UniFormer":
        image_root = model_root / "image_classification"
        sys.path.insert(0, str(image_root))
        module = importlib.import_module("models.uniformer")
        base = module.uniformer_small(pretrained=False, img_size=image_size, num_classes=0, drop_path_rate=0.1)
        base.norm = nn.Identity()
        base.pre_logits = nn.Identity()
        base.head = nn.Identity()
        return UniFormerFeatures(base), [64, 128, 320, 512], "UniFormer-Small", flop_handles
    raise ValueError(f"Unknown model: {model_name}")


def build_detector(model_name: str, model_root: Path, image_size: int) -> Tuple[nn.Module, str]:
    body, channels, variant, flop_handles = build_feature_body(model_name, model_root, image_size)
    backbone = BackboneFPN(body, channels, out_channels=128)
    bases = (16, 32, 64, 128, 256)
    sizes = tuple(tuple(int(round(x * (2 ** (k / 3)))) for k in range(3)) for x in bases)
    ratios = tuple((0.5, 1.0, 2.0) for _ in bases)
    anchors = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
    detector = RetinaNet(
        backbone=backbone,
        num_classes=len(CLASS_NAMES) + 1,
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
    detector._benchmark_flop_handles = flop_handles
    detector._benchmark_variant = variant
    return detector, variant


def box_iou_numpy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    values = []
    for threshold in np.linspace(0.0, 1.0, 101):
        valid = precision[recall >= threshold]
        values.append(float(valid.max()) if valid.size else 0.0)
    return float(np.mean(values))


def match_class(predictions, gt_by_image, npos: int, iou_threshold: float, score_cutoff: float = -1.0):
    matched = defaultdict(set)
    true_pos, false_pos = [], []
    for image_id, score, box in predictions:
        if score < score_cutoff:
            continue
        gt_boxes = gt_by_image.get(image_id, np.empty((0, 4), dtype=np.float32))
        ious = box_iou_numpy(box, gt_boxes)
        best = int(ious.argmax()) if ious.size else -1
        if best >= 0 and ious[best] >= iou_threshold and best not in matched[image_id]:
            matched[image_id].add(best)
            true_pos.append(1.0)
            false_pos.append(0.0)
        else:
            true_pos.append(0.0)
            false_pos.append(1.0)
    tp = np.asarray(true_pos, dtype=np.float64)
    fp = np.asarray(false_pos, dtype=np.float64)
    if tp.size:
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
    recall = tp / max(npos, 1)
    precision = tp / np.maximum(tp + fp, 1e-12)
    return precision, recall, (float(tp[-1]) if tp.size else 0.0), (float(fp[-1]) if fp.size else 0.0)


def detection_metrics(records: List[dict], num_classes: int = 7) -> dict:
    thresholds = [0.50 + 0.05 * i for i in range(10)]
    aps_by_threshold = [[] for _ in thresholds]
    macro_precision, macro_recall = [], []
    per_class = {}
    for class_id in range(1, num_classes + 1):
        gt_by_image = {}
        predictions = []
        for record in records:
            gt_mask = record["gt_labels"] == class_id
            if gt_mask.any():
                gt_by_image[record["image_id"]] = record["gt_boxes"][gt_mask]
            pred_mask = record["pred_labels"] == class_id
            for box, score in zip(record["pred_boxes"][pred_mask], record["pred_scores"][pred_mask]):
                predictions.append((record["image_id"], float(score), box))
        predictions.sort(key=lambda item: item[1], reverse=True)
        npos = sum(len(boxes) for boxes in gt_by_image.values())
        if npos == 0:
            continue
        class_aps = []
        for threshold_index, threshold in enumerate(thresholds):
            precision, recall, _, _ = match_class(predictions, gt_by_image, npos, threshold)
            ap = ap_101(recall, precision)
            aps_by_threshold[threshold_index].append(ap)
            class_aps.append(ap)
        _, _, tp, fp = match_class(predictions, gt_by_image, npos, 0.50, score_cutoff=0.25)
        class_precision = tp / max(tp + fp, 1e-12)
        class_recall = tp / max(npos, 1)
        macro_precision.append(class_precision)
        macro_recall.append(class_recall)
        per_class[CLASS_NAMES[class_id - 1]] = {
            "precision": class_precision,
            "recall": class_recall,
            "AP50": class_aps[0],
            "AP50_95": float(np.mean(class_aps)),
            "ground_truths": npos,
        }
    map_by_threshold = [float(np.mean(values)) if values else 0.0 for values in aps_by_threshold]
    return {
        "precision": float(np.mean(macro_precision)) if macro_precision else 0.0,
        "recall": float(np.mean(macro_recall)) if macro_recall else 0.0,
        "mAP50": map_by_threshold[0],
        "mAP50_95": float(np.mean(map_by_threshold)),
        "AP_by_IoU": {f"{threshold:.2f}": value for threshold, value in zip(thresholds, map_by_threshold)},
        "per_class": per_class,
        "precision_recall_definition": "macro class mean at score>=0.25 and IoU=0.50",
        "ap_definition": "COCO-style 101-point interpolated AP",
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True, max_batches: int = 0) -> dict:
    model.eval()
    records = []
    for batch_index, (images, targets) in enumerate(loader):
        gpu_images = [image.to(device, non_blocking=True) for image in images]
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            outputs = model(gpu_images)
        for target, output in zip(targets, outputs):
            records.append({
                "image_id": int(target["image_id"].item()),
                "gt_boxes": target["boxes"].numpy(),
                "gt_labels": target["labels"].numpy(),
                "pred_boxes": output["boxes"].detach().cpu().numpy(),
                "pred_labels": output["labels"].detach().cpu().numpy(),
                "pred_scores": output["scores"].detach().cpu().numpy(),
            })
        if max_batches and batch_index + 1 >= max_batches:
            break
    metrics = detection_metrics(records, num_classes=len(CLASS_NAMES))
    metrics["images"] = len(records)
    return metrics


def train_one_epoch(model, loader, optimizer, scaler, device, amp: bool, max_batches: int = 0):
    model.train()
    totals = defaultdict(float)
    batches = 0
    for batch_index, (images, targets) in enumerate(loader):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at batch {batch_index}: {loss_dict}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        totals["train_loss"] += float(loss.detach())
        for key, value in loss_dict.items():
            totals[f"loss_{key}"] += float(value.detach())
        batches += 1
        if max_batches and batch_index + 1 >= max_batches:
            break
    return {key: value / max(batches, 1) for key, value in totals.items()}


class RawDetectorForward(nn.Module):
    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector

    def forward(self, x: Tensor):
        features = self.detector.backbone(x)
        outputs = self.detector.head(list(features.values()))
        return outputs["cls_logits"], outputs["bbox_regression"]


@torch.no_grad()
def profile_detector(model: nn.Module, device: torch.device, image_size: int, seed: int) -> dict:
    result = {
        "Params_M": sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        "input": [1, 3, image_size, image_size],
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "fps_scope": "batch-1 end-to-end RetinaNet forward including transform, decoding and NMS",
        "flops_scope": "backbone + FPN + RetinaNet classification/regression heads; excludes anchors, decoding and NMS",
        "seed_checkpoint": seed,
    }
    sample = torch.randn(1, 3, image_size, image_size, device=device)
    raw = RawDetectorForward(model).eval()
    try:
        from fvcore.nn import FlopCountAnalysis
        analysis = FlopCountAnalysis(raw, sample)
        handles = getattr(model, "_benchmark_flop_handles", {})
        for op_name, handle in handles.items():
            analysis = analysis.set_op_handle(op_name, handle)
        flops = float(analysis.total())
        result["FLOPs_G"] = flops / 1e9
        result["flops_tool"] = "fvcore FlopCountAnalysis (one fused multiply-add counted as one operation)"
        result["unsupported_ops"] = {str(k): int(v) for k, v in analysis.unsupported_ops().items()}
    except Exception as exc:
        result["FLOPs_G"] = None
        result["flops_error"] = f"{type(exc).__name__}: {exc}"
    if device.type == "cuda":
        model.eval()
        image = torch.randn(3, image_size, image_size, device=device)
        for _ in range(20):
            model([image])
        torch.cuda.synchronize()
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            model([image])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        result["FPS"] = iterations / elapsed
        result["latency_ms"] = elapsed * 1000.0 / iterations
        result["fps_warmup"] = 20
        result["fps_iterations"] = iterations
    else:
        result["FPS"] = None
    return result


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def append_metrics(path: Path, row: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in METRIC_FIELDS})
        f.flush()


def read_best_row(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return max(rows, key=lambda row: float(row["mAP50_95"])) if rows else {}


def run_smoke(args, model, train_loader, val_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    train_values = train_one_epoch(model, train_loader, optimizer, scaler, device, args.amp, max_batches=1)
    val_values = evaluate(model, val_loader, device, args.amp, max_batches=1)
    print(json.dumps({"smoke": "ok", "model": args.model, "train": train_values, "val": val_values}, ensure_ascii=False), flush=True)


def run_training(args) -> None:
    model_root = Path(__file__).resolve().parent
    dataset_root = Path(args.dataset) if args.dataset else model_root.parent / "datasets" / "m2cai16-tool-locations"
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is required for this benchmark; use --allow-cpu only for diagnostics")
    seed_everything(args.seed)
    print(f"Building {args.model} on {device} from {model_root}", flush=True)
    model, variant = build_detector(args.model, model_root, args.image_size)
    model.to(device)
    train_set = VocToolDataset(dataset_root, "train", args.image_size, augment=True)
    val_set = VocToolDataset(dataset_root, "val", args.image_size, augment=False)
    test_set = VocToolDataset(dataset_root, "test", args.image_size, augment=False)
    if args.smoke:
        train_set = Subset(train_set, list(range(min(args.batch_size, len(train_set)))))
        val_set = Subset(val_set, [0])
    train_loader = make_loader(train_set, args.batch_size, args.workers, True, args.seed)
    val_loader = make_loader(val_set, args.eval_batch_size, args.workers, False, args.seed + 1)
    if args.smoke:
        run_smoke(args, model, train_loader, val_loader, device)
        return

    run_dir = model_root / "runs" / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    config = {
        "model": args.model,
        "variant": variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "eta_min": args.eta_min,
        "amp": args.amp,
        "gradient_clip_norm": 5.0,
        "detector": "torchvision RetinaNet + 128-channel FPN",
        "pretrained": False,
        "dataset": str(dataset_root),
        "splits": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "classes": list(CLASS_NAMES),
        "torch": torch.__version__,
        "torchvision": importlib.import_module("torchvision").__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    atomic_json(run_dir / "config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    start_epoch, best_map = 1, -1.0
    last_path = run_dir / "last.pt"
    if args.resume and last_path.exists():
        checkpoint = torch_load_path(last_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map = float(checkpoint.get("best_mAP50_95", -1.0))
        print(f"Resuming at epoch {start_epoch}", flush=True)
    elif metrics_path.exists():
        recovered = metrics_path.with_name(f"metrics_incomplete_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        os.replace(metrics_path, recovered)
        print(f"Archived incomplete metrics as {recovered.name}", flush=True)

    status = {"state": "RUNNING", "model": args.model, "seed": args.seed, "epoch": start_epoch - 1, "epochs": args.epochs}
    atomic_json(run_dir / "status.json", status)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_values = train_one_epoch(model, train_loader, optimizer, scaler, device, args.amp)
            val_values = evaluate(model, val_loader, device, args.amp)
            elapsed = time.perf_counter() - epoch_start
            row = {
                "epoch": epoch,
                "train_loss": train_values.get("train_loss", 0.0),
                "loss_classification": train_values.get("loss_classification", 0.0),
                "loss_bbox_regression": train_values.get("loss_bbox_regression", 0.0),
                "precision": val_values["precision"],
                "recall": val_values["recall"],
                "mAP50": val_values["mAP50"],
                "mAP50_95": val_values["mAP50_95"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": elapsed,
                "val_images": val_values["images"],
            }
            append_metrics(metrics_path, row)
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
            atomic_torch_save(last_path, checkpoint)
            if val_values["mAP50_95"] > best_map:
                best_map = val_values["mAP50_95"]
                checkpoint["best_mAP50_95"] = best_map
                atomic_torch_save(run_dir / "best.pt", checkpoint)
                atomic_json(run_dir / "best_val_metrics.json", {"epoch": epoch, **val_values})
            status.update({"epoch": epoch, "last_metrics": row, "best_mAP50_95": best_map, "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
            atomic_json(run_dir / "status.json", status)
            print(json.dumps({"model": args.model, "seed": args.seed, **row}, ensure_ascii=False), flush=True)

        best_path = run_dir / "best.pt"
        checkpoint = torch_load_path(best_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        test_loader = make_loader(test_set, args.eval_batch_size, args.workers, False, args.seed + 2)
        test_values = evaluate(model, test_loader, device, args.amp)
        atomic_json(run_dir / "test_metrics.json", test_values)
        profile = profile_detector(model, device, args.image_size, args.seed)
        atomic_json(run_dir / "profile.json", profile)
        if args.seed == 42:
            atomic_json(model_root / "profile.json", profile)
        best_row = read_best_row(metrics_path)
        summary = {
            "model": args.model,
            "variant": variant,
            "seed": args.seed,
            "best_validation": best_row,
            "test": test_values,
            "profile": profile,
        }
        atomic_json(run_dir / "summary.json", summary)
        status.update({"state": "COMPLETED", "test": test_values, "profile": profile, "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
        atomic_json(run_dir / "status.json", status)
    except Exception as exc:
        status.update({"state": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
        atomic_json(run_dir / "status.json", status)
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--dataset", default="")
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
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
