#!/usr/bin/env python
"""Pre-training structural, initialization, forward and backward checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import train_m2cai as benchmark
from ser_cmt import DALFE, ECSA, SERBlock, architecture_manifest, ecsa_modules
from train_ser_cmt import build_detector, load_dalfe_initialization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--device", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--dalfe-checkpoint", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    donor = Path(args.dalfe_checkpoint) if args.dalfe_checkpoint else None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    benchmark.seed_everything(42)
    model, base = build_detector(args.image_size, use_valid_mask=True)
    load_report = load_dalfe_initialization(model, donor)
    model.to(device)

    manifest = architecture_manifest(base)
    expected_shapes = [
        [1, 46, args.image_size // 4, args.image_size // 4],
        [1, 92, args.image_size // 8, args.image_size // 8],
        [1, 184, args.image_size // 16, args.image_size // 16],
        [1, 368, args.image_size // 32, args.image_size // 32],
    ]
    model.eval()
    with torch.no_grad():
        features = model.backbone.body(torch.randn(1, 3, args.image_size, args.image_size, device=device))
        observed_shapes = [list(feature.shape) for feature in features]
        outputs = model([torch.rand(3, args.image_size, args.image_size, device=device)])
    if observed_shapes != expected_shapes:
        raise AssertionError(f"Stage shape mismatch: {observed_shapes} != {expected_shapes}")
    if len(outputs) != 1 or not {"boxes", "labels", "scores"}.issubset(outputs[0]):
        raise AssertionError("Detector evaluation forward did not return standard predictions")

    ser_indices = [index for index, block in enumerate(base.blocks_c) if isinstance(block, SERBlock)]
    if ser_indices != list(range(10)):
        raise AssertionError(f"All ten CMT-Ti Stage3 blocks must be SER blocks: {ser_indices}")
    if any(not isinstance(base.blocks_c[index].proj, DALFE) for index in ser_indices):
        raise AssertionError("One or more Stage3 SER blocks do not contain DALFE")
    if any(not isinstance(base.blocks_c[index].attn, ECSA) for index in ser_indices):
        raise AssertionError("One or more Stage3 SER blocks do not contain ECSA")
    for _, module in ecsa_modules(base):
        if float(module.eta.detach()) != 0.0 or float(module.gamma.detach()) != 0.0:
            raise AssertionError("ECSA eta/gamma must initialize at exactly zero")

    # One real detector loss backward verifies gradients through the new path.
    model.train()
    image = torch.rand(3, args.image_size, args.image_size, device=device)
    target = {
        "boxes": torch.tensor([[60.0, 80.0, 230.0, 200.0]], device=device),
        "labels": torch.tensor([1], dtype=torch.int64, device=device),
    }
    loss_dict = model([image], [target])
    loss = sum(loss_dict.values())
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if not torch.isfinite(loss) or not finite_gradients:
        raise AssertionError("Non-finite loss or gradient in backward check")

    edge = base.semantic_edge_prior
    alpha, beta = edge.fusion_weights()
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report = {
        "status": "PASS",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "stage_shapes_expected": expected_shapes,
        "stage_shapes_observed": observed_shapes,
        "stage3_ser_indices_zero_based": ser_indices,
        "forward_prediction_counts": {key: len(value) for key, value in outputs[0].items()},
        "backward_loss": float(loss.detach()),
        "backward_loss_components": {key: float(value.detach()) for key, value in loss_dict.items()},
        "finite_gradients": bool(finite_gradients),
        "params": params,
        "params_M": params / 1e6,
        "trainable_params": trainable,
        "alpha_initial": alpha,
        "beta_initial": beta,
        "eta_initial": [float(module.eta.detach()) for _, module in ecsa_modules(base)],
        "gamma_initial": [float(module.gamma.detach()) for _, module in ecsa_modules(base)],
        "weight_loading": load_report,
        "architecture": manifest,
    }
    output = Path(args.output) if args.output else root / "preflight_validation.json"
    benchmark.atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
