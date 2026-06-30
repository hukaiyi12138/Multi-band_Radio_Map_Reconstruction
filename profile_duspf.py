#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


# =========================
# 路径配置
# =========================
ROOT = Path("/data/home/hky/DULRTC")
MODEL_DIR = ROOT / "DUSPF-RME"
OUT_PATH = MODEL_DIR / "DUSPF-RME_profile.txt"

sys.path.insert(0, str(MODEL_DIR))

from model import DUSPF_RME


# =========================
# 模型配置：按你的真实实验设置改这里
# =========================
R = 3
K = 3
H = 256
W = 256
N_ITER = 10

PROX_HIDDEN = 48
PROX_LAYERS = 3
N_RAY_SAMPLES = 32
GN_GROUPS = 8
PROX_INIT_STD = 0.001
SKIP_LAST_PROX = True
PROX_BACKBONE = "unet"

BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP = 10
RUNS = 100


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def fmt_m(n: int) -> str:
    return f"{n / 1e6:.4f} M"


def build_model():
    model = DUSPF_RME(
        R=R,
        K=K,
        N_iter=N_ITER,
        prox_hidden=PROX_HIDDEN,
        prox_layers=PROX_LAYERS,
        n_ray_samples=N_RAY_SAMPLES,
        gn_groups=GN_GROUPS,
        prox_init_std=PROX_INIT_STD,
        skip_last_prox=SKIP_LAST_PROX,
        prox_backbone=PROX_BACKBONE,
    )
    return model


def build_dummy_inputs(device):
    """
    根据 DUSPF-RME 的常见 forward 输入构造 dummy data。
    如果你的 forward 输入顺序不一样，只需要改这里。
    """
    D_obs = torch.rand(BATCH_SIZE, K, H, W, device=device)
    Omega = torch.randint(0, 2, (BATCH_SIZE, K, H, W), device=device).float()
    B = torch.randint(0, 2, (BATCH_SIZE, 1, H, W), device=device).float()

    # transmitter positions: [B, R, 2]，坐标归一化到 [0, 1]
    tx_pos = torch.rand(BATCH_SIZE, R, 2, device=device)

    return D_obs, Omega, B, tx_pos


def forward_once(model, inputs):
    """
    这里兼容几种常见 forward 写法。
    如果都失败，报错里会提示你该改这里。
    """
    D_obs, Omega, B, tx_pos = inputs

    try:
        return model(D_obs, Omega, B, tx_pos)
    except TypeError:
        pass

    try:
        return model(D_obs=D_obs, Omega=Omega, B=B, tx_pos=tx_pos)
    except TypeError:
        pass

    try:
        return model(D_obs, Omega, B)
    except TypeError:
        pass

    try:
        return model(D_obs=D_obs, Omega=Omega, B=B)
    except TypeError:
        pass

    raise RuntimeError(
        "Forward call failed. Please edit forward_once() according to your model.forward signature."
    )


@torch.no_grad()
def measure_inference_time(model, inputs, device):
    model.eval()

    # warm-up
    for _ in range(WARMUP):
        _ = forward_once(model, inputs)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    for _ in range(RUNS):
        _ = forward_once(model, inputs)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    return (end - start) * 1000.0 / RUNS


def main():
    model = build_model().to(DEVICE)
    inputs = build_dummy_inputs(DEVICE)

    total, trainable, non_trainable = count_params(model)

    try:
        time_ms = measure_inference_time(model, inputs, DEVICE)
        time_msg = f"{time_ms:.4f} ms / sample"
    except Exception as e:
        time_ms = None
        time_msg = f"FAILED: {repr(e)}"

    lines = []
    lines.append("=" * 80)
    lines.append("DUSPF-RME Parameter Count and Inference Time")
    lines.append("=" * 80)
    lines.append(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    lines.append("")
    lines.append("Model configuration:")
    lines.append(f"R = {R}")
    lines.append(f"K = {K}")
    lines.append(f"H = {H}")
    lines.append(f"W = {W}")
    lines.append(f"N_iter = {N_ITER}")
    lines.append(f"prox_hidden = {PROX_HIDDEN}")
    lines.append(f"prox_layers = {PROX_LAYERS}")
    lines.append(f"n_ray_samples = {N_RAY_SAMPLES}")
    lines.append(f"gn_groups = {GN_GROUPS}")
    lines.append(f"prox_init_std = {PROX_INIT_STD}")
    lines.append(f"skip_last_prox = {SKIP_LAST_PROX}")
    lines.append(f"prox_backbone = {PROX_BACKBONE}")
    lines.append("")
    lines.append("Parameter count:")
    lines.append(f"Total parameters:         {total} ({fmt_m(total)})")
    lines.append(f"Trainable parameters:     {trainable} ({fmt_m(trainable)})")
    lines.append(f"Non-trainable parameters: {non_trainable} ({fmt_m(non_trainable)})")
    lines.append("")
    lines.append("Inference time:")
    lines.append(f"Batch size: {BATCH_SIZE}")
    lines.append(f"Warm-up runs: {WARMUP}")
    lines.append(f"Measured runs: {RUNS}")
    lines.append(f"Average time: {time_msg}")
    lines.append("")
    lines.append("Top-level module breakdown:")
    lines.append("")
    lines.append("Proximal module breakdown inside each block:")
    lines.append("-" * 80)
    lines.append(f"{'Block':15s} {'Submodule':25s} {'Trainable Params':>18s}")
    lines.append("-" * 80)

    for i, block in enumerate(model.blocks):
        for name, module in block.named_children():
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if n > 0:
                lines.append(f"{'block_' + str(i):15s} {name:25s} {fmt_m(n):>18s}")

    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
