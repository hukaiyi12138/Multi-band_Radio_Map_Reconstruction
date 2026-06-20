"""
test.py - Evaluation script for PR-BTD-DULRTC on DULRTC_triple/Test_Samples.

Output style:

output_path/
└── Mask15/
    ├── metrics_01.txt
    ├── metrics_01.json
    └── figures/
        ├── 893/
        │   ├── 893_gt_c0.png
        │   ├── 893_gt_c1.png
        │   ├── 893_gt_c2.png
        │   ├── 893_mask15.png
        │   ├── 893_mask15_pred_c0.png
        │   ├── 893_mask15_pred_c1.png
        │   └── 893_mask15_pred_c2.png
        └── ...

Recommended command:

CUDA_VISIBLE_DEVICES=0 python test.py \
    --checkpoint runs/fiber1/best.pt \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 1 \
    --mask-type fiber \
    --R 3 \
    --K 3 \
    --N-iter 10 \
    --batch-size 1 \
    --output-path /data/home/hky/DULRTC/hky_try_3/test \
    --max-save-figures 10
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import PR_BTD_DULRTC
from util import DULRTCTripleDataset, SyntheticBTDDataset


# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint", required=True)

    # dataset
    p.add_argument(
        "--dataset",
        choices=["dulrtc_triple", "synthetic"],
        default="dulrtc_triple",
    )
    p.add_argument(
        "--root",
        type=str,
        default="/data/home/hky/dataset/DULRTC_triple",
    )
    p.add_argument(
        "--omega-num",
        type=int,
        default=15,
        choices=[1, 5, 10, 15],
    )
    p.add_argument(
        "--mask-type",
        choices=["mask", "fiber"],
        default="mask",
    )

    # model / data shape
    p.add_argument("--R", type=int, default=3)
    p.add_argument("--K", type=int, default=3)

    p.add_argument(
        "--N-iter",
        type=int,
        default=10,
        help="Fallback N_iter if checkpoint does not store N_iter. Default: 10.",
    )

    # evaluation
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true")

    p.add_argument(
        "--output-path",
        type=str,
        default="test",
        help="Root test output directory.",
    )
    p.add_argument(
        "--out-threshold",
        type=float,
        default=0.4,
        help="Threshold for outage map.",
    )
    p.add_argument(
        "--no-save-figures",
        action="store_true",
        help="Disable PNG visualisation saving.",
    )
    p.add_argument(
        "--max-save-figures",
        type=int,
        default=10,
        help=(
            "Maximum number of test samples to save figures for. "
            "Default is 10. Set <=0 to save figures for all samples."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
def build_exp_name(mask_type: str, omega_num: int) -> str:
    prefix = "Fiber" if mask_type.lower() == "fiber" else "Mask"
    return f"{prefix}{omega_num}"


# ---------------------------------------------------------------------------
def build_file_tag(mask_type: str, omega_num: int) -> str:
    prefix = "fiber" if mask_type.lower() == "fiber" else "mask"
    return f"{prefix}{omega_num}"


# ---------------------------------------------------------------------------
def to_jsonable(obj):
    """
    Convert numpy scalar / ndarray objects to JSON-serializable Python types.
    This avoids:
        TypeError: Object of type float32 is not JSON serializable
    """
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    return obj


# ---------------------------------------------------------------------------
def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
def make_dataset(args, split: str = "test"):
    if args.dataset == "synthetic":
        return SyntheticBTDDataset(
            n_samples=64,
            H=64,
            W=64,
            K=args.K,
            R=args.R,
            omega_ratio=0.1,
        )

    if args.dataset == "dulrtc_triple":
        return DULRTCTripleDataset(
            root=args.root,
            split=split,
            omega_num=args.omega_num,
            mask_type=args.mask_type,
            R=args.R,
            K=args.K,
        )

    raise ValueError(f"Unknown dataset: {args.dataset}")


# ---------------------------------------------------------------------------
def collate(batch):
    Ds = torch.stack([b[0] for b in batch])
    Oms = torch.stack([b[1] for b in batch])
    Bs = torch.stack([b[2] for b in batch])
    Tx = torch.stack([b[3] for b in batch])
    metas = [b[4] for b in batch]
    return Ds, Oms, Bs, Tx, metas


# ---------------------------------------------------------------------------
def get_ck_arg(ck_args: dict, names, default=None):
    """
    Read checkpoint args robustly.
    Compatible with different training script key names.
    """
    for name in names:
        if name in ck_args and ck_args[name] is not None:
            return ck_args[name]
    return default


# ---------------------------------------------------------------------------
def load_model(args, device):
    ck_path = Path(args.checkpoint)

    if not ck_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}")

    ck = torch.load(str(ck_path), map_location=device)
    ck_args = ck.get("args", {})

    n_iter = get_ck_arg(
        ck_args,
        names=["N_iter", "N-iter", "n_iter", "n_iters"],
        default=args.N_iter,
    )

    model = PR_BTD_DULRTC(
        R=int(ck_args.get("R", args.R)),
        K=int(ck_args.get("K", args.K)),
        N_iter=int(n_iter),
        prox_hidden=int(ck_args.get("prox_hidden", 32)),
        prox_layers=int(ck_args.get("prox_layers", 3)),
        n_ray_samples=int(ck_args.get("n_ray_samples", 16)),
    ).to(device)

    state = ck["model"]

    # Compatible with DataParallel checkpoints.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()

    print(f"[setup] loaded checkpoint: {ck_path}")
    print(f"[setup] checkpoint epoch: {ck.get('epoch', 'unknown')}")
    print(f"[setup] checkpoint best_val: {ck.get('best_val', 'unknown')}")
    print(f"[setup] model R={model.R}, K={model.K}, N_iter={int(n_iter)}")

    return model, int(n_iter)


# ---------------------------------------------------------------------------
def inference(model, D, Om, B, Tx):
    start = time.time()

    with torch.no_grad():
        D_obs = D * Om
        D_hat, X, E, S, c = model(D_obs, Om, B, Tx)

    elapsed = time.time() - start
    return D_hat, X, E, S, c, elapsed


# ---------------------------------------------------------------------------
def cal_PSNR(X, X_hat):
    mse = np.mean((X - X_hat) ** 2)
    if mse == 0:
        return 100.0

    max_val = np.max(X_hat)
    if max_val <= 0:
        max_val = 1.0

    return 20.0 * np.log10(max_val / np.sqrt(mse))


# ---------------------------------------------------------------------------
def cal_NMSE(X, X_hat):
    denom = np.sum(X ** 2)
    if denom == 0:
        return 0.0
    return np.sum((X - X_hat) ** 2) / denom


# ---------------------------------------------------------------------------
def cal_RMSE(X, X_hat):
    mse = np.mean((X - X_hat) ** 2)
    return np.sqrt(mse)


# ---------------------------------------------------------------------------
def create_outage_map(X, threshold):
    return (X >= threshold).astype(np.float32)


# ---------------------------------------------------------------------------
def cal_seg_error(X, X_hat, threshold):
    X_map = create_outage_map(X, threshold)
    X_hat_map = create_outage_map(X_hat, threshold)
    mse = np.mean((X_map - X_hat_map) ** 2)
    rmse = np.sqrt(mse)
    return mse, rmse


# ---------------------------------------------------------------------------
def khw_to_hwc(arr):
    """
    [K,H,W] -> [H,W,K]
    """
    arr = np.asarray(arr).astype(np.float32)

    if arr.ndim == 2:
        return arr[:, :, None]

    if arr.ndim == 3:
        return np.transpose(arr, (1, 2, 0))

    raise ValueError(f"Unsupported shape: {arr.shape}")


# ---------------------------------------------------------------------------
def save_viridis_png(img_2d, save_path, vmin=None, vmax=None, white_mask=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed - skipping visualisation.")
        return

    img_2d = np.asarray(img_2d).astype(np.float32)

    cmap = plt.get_cmap("viridis").copy()
    vis = np.array(img_2d, dtype=np.float32, copy=True)

    if white_mask is not None:
        cmap.set_bad(color="white")
        vis = np.ma.array(vis, mask=white_mask.astype(bool))

    plt.figure(figsize=(5, 5))
    plt.imshow(vis, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


# ---------------------------------------------------------------------------
def save_binary_mask_png(mask_2d, save_path):
    """
    Save sampling mask:
        sampled pixels   -> black
        unsampled pixels -> white
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed - skipping mask visualisation.")
        return

    mask_2d = np.asarray(mask_2d).astype(np.float32)
    sampled = mask_2d > 0

    # sampled = 0 black, missing = 1 white
    vis = np.ones_like(mask_2d, dtype=np.float32)
    vis[sampled] = 0.0

    plt.figure(figsize=(5, 5))
    plt.imshow(vis, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


# ---------------------------------------------------------------------------
def save_channel_figures(
    gt_khw,
    pred_khw,
    omega_khw,
    map_fig_dir: Path,
    map_id: str,
    file_tag: str,
):
    """
    Save per-channel GT and prediction figures.

    gt_khw    : [K,H,W]
    pred_khw  : [K,H,W]
    omega_khw : [K,H,W]

    Output:
        {map_id}_gt_c0.png
        {map_id}_gt_c1.png
        {map_id}_gt_c2.png
        {map_id}_{file_tag}.png
        {map_id}_{file_tag}_pred_c0.png
        {map_id}_{file_tag}_pred_c1.png
        {map_id}_{file_tag}_pred_c2.png
    """
    gt_khw = np.asarray(gt_khw).astype(np.float32)
    pred_khw = np.asarray(pred_khw).astype(np.float32)
    omega_khw = np.asarray(omega_khw).astype(np.float32)

    K = gt_khw.shape[0]

    # One global visual range for all channels in this sample.
    vmin = float(min(gt_khw.min(), pred_khw.min()))
    vmax = float(max(gt_khw.max(), pred_khw.max()))

    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8

    # Save one sampling mask image.
    # If Omega differs across channels, use union of observed pixels.
    obs_mask = np.sum(omega_khw > 0, axis=0) > 0
    save_binary_mask_png(
        obs_mask.astype(np.float32),
        map_fig_dir / f"{map_id}_{file_tag}.png",
    )

    # Save channel-wise GT and prediction.
    for c in range(K):
        save_viridis_png(
            gt_khw[c],
            map_fig_dir / f"{map_id}_gt_c{c}.png",
            vmin=vmin,
            vmax=vmax,
            white_mask=None,
        )

        save_viridis_png(
            pred_khw[c],
            map_fig_dir / f"{map_id}_{file_tag}_pred_c{c}.png",
            vmin=vmin,
            vmax=vmax,
            white_mask=None,
        )


# ---------------------------------------------------------------------------
def format_metrics_table(rows, avg_row):
    headers = ["sample", "PSNR", "RMSE", "NMSE", "outage_mse", "outage_rmse"]

    col_widths = []

    for col_idx in range(len(headers)):
        max_len = len(headers[col_idx])
        max_len = max(max_len, len(avg_row[col_idx]))

        for row in rows:
            max_len = max(max_len, len(row[col_idx]))

        col_widths.append(max_len + 2)

    def format_row(row):
        return "".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    lines = []
    lines.append(format_row(headers))
    lines.append(format_row(avg_row))
    lines.append(format_row(["-" * (w - 2) for w in col_widths]))

    for row in rows:
        lines.append(format_row(row))

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
def main():
    args = get_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"CUDA visible device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    exp_name = build_exp_name(args.mask_type, args.omega_num)
    file_tag = build_file_tag(args.mask_type, args.omega_num)

    final_output_path = Path(args.output_path) / exp_name
    final_output_path.mkdir(parents=True, exist_ok=True)

    figures_root = final_output_path / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    metrics_path = final_output_path / "metrics_01.txt"
    metrics_json_path = final_output_path / "metrics_01.json"

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output path: {final_output_path}")

    if args.no_save_figures:
        print("[setup] figure saving disabled.")
    else:
        if args.max_save_figures <= 0:
            print("[setup] saving figures for all test samples.")
        else:
            print(f"[setup] saving figures for first {args.max_save_figures} test samples.")

    model, n_iter = load_model(args, device)

    ds = make_dataset(args, split="test")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[setup] dataset: {args.dataset}")
    print(f"[setup] test samples: {len(ds)}")
    print(f"[setup] omega: {args.mask_type.upper()}_{args.omega_num}")

    rows = []

    PSNR_sum = 0.0
    RMSE_sum = 0.0
    NMSE_sum = 0.0
    outage_mse_sum = 0.0
    outage_rmse_sum = 0.0
    total_count = 0
    figure_count = 0

    for batch_idx, (D, Om, B, Tx, metas) in enumerate(loader):
        D = D.to(device, non_blocking=True)
        Om = Om.to(device, non_blocking=True)
        B = B.to(device, non_blocking=True)
        Tx = Tx.to(device, non_blocking=True)

        try:
            D_hat, X, E, S, c, elapsed = inference(model, D, Om, B, Tx)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[warn] CUDA OOM on batch {batch_idx}; skipping.")
                clear_cuda_cache()
                continue
            raise e

        D_hat = torch.clamp(D_hat, 0.0, 1.0)

        D_cpu = D.detach().cpu().numpy()
        Om_cpu = Om.detach().cpu().numpy()
        D_hat_cpu = D_hat.detach().cpu().numpy()

        batch_size = D_cpu.shape[0]

        for i in range(batch_size):
            meta = metas[i]
            sample_name = meta.get("sample_id", f"sample_{batch_idx}_{i}")
            map_id = Path(sample_name).stem

            print("Processing set " + str(total_count + 1).zfill(6))

            gt_khw = D_cpu[i]          # [K,H,W]
            pred_khw = D_hat_cpu[i]    # [K,H,W]
            omega_khw = Om_cpu[i]      # [K,H,W]

            gt_hwc = khw_to_hwc(gt_khw)
            pred_hwc = khw_to_hwc(pred_khw)

            save_this_figure = (
                (not args.no_save_figures)
                and (args.max_save_figures <= 0 or total_count < args.max_save_figures)
            )

            if save_this_figure:
                map_fig_dir = figures_root / map_id
                map_fig_dir.mkdir(parents=True, exist_ok=True)

                save_channel_figures(
                    gt_khw=gt_khw,
                    pred_khw=pred_khw,
                    omega_khw=omega_khw,
                    map_fig_dir=map_fig_dir,
                    map_id=map_id,
                    file_tag=file_tag,
                )
                figure_count += 1

            PSNR = cal_PSNR(gt_hwc, pred_hwc)
            RMSE = cal_RMSE(gt_hwc, pred_hwc)
            NMSE = cal_NMSE(gt_hwc, pred_hwc)

            outage_mse, outage_rmse = cal_seg_error(
                gt_hwc,
                pred_hwc,
                args.out_threshold,
            )

            PSNR_sum += float(PSNR)
            RMSE_sum += float(RMSE)
            NMSE_sum += float(NMSE)
            outage_mse_sum += float(outage_mse)
            outage_rmse_sum += float(outage_rmse)
            total_count += 1

            print(f"Cost {elapsed:.4f} s")
            print(
                f"[{sample_name}] "
                f"PSNR: {PSNR:.4f}, "
                f"RMSE: {RMSE:.4f}, "
                f"NMSE: {NMSE:.4f}, "
                f"outage_mse: {outage_mse:.4f}, "
                f"outage_rmse: {outage_rmse:.4f}"
            )

            rows.append(
                [
                    f"[{sample_name}]",
                    f"{PSNR:.4f}",
                    f"{RMSE:.4f}",
                    f"{NMSE:.4f}",
                    f"{outage_mse:.4f}",
                    f"{outage_rmse:.4f}",
                ]
            )

        del D, Om, B, Tx, D_hat, X, E, S, c
        clear_cuda_cache()

    if total_count == 0:
        raise RuntimeError("No valid test samples were processed.")

    PSNR_avg = float(PSNR_sum / total_count)
    RMSE_avg = float(RMSE_sum / total_count)
    NMSE_avg = float(NMSE_sum / total_count)
    outage_mse_avg = float(outage_mse_sum / total_count)
    outage_rmse_avg = float(outage_rmse_sum / total_count)

    print("Average results:")
    print(
        f"PSNR: {PSNR_avg:.4f}, "
        f"RMSE: {RMSE_avg:.4f}, "
        f"NMSE: {NMSE_avg:.4f}, "
        f"outage_mse: {outage_mse_avg:.4f}, "
        f"outage_rmse: {outage_rmse_avg:.4f}"
    )

    avg_row = [
        "[average]",
        f"{PSNR_avg:.4f}",
        f"{RMSE_avg:.4f}",
        f"{NMSE_avg:.4f}",
        f"{outage_mse_avg:.4f}",
        f"{outage_rmse_avg:.4f}",
    ]

    table_text = format_metrics_table(rows, avg_row)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(table_text)

    summary = {
        "average": {
            "PSNR": float(PSNR_avg),
            "RMSE": float(RMSE_avg),
            "NMSE": float(NMSE_avg),
            "outage_mse": float(outage_mse_avg),
            "outage_rmse": float(outage_rmse_avg),
        },
        "n_samples": int(total_count),
        "out_threshold": float(args.out_threshold),
        "max_save_figures": int(args.max_save_figures),
        "saved_figure_samples": int(figure_count),
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "split": "Test_Samples",
        "omega": f"{args.mask_type.upper()}_{args.omega_num}",
        "N_iter": int(n_iter),
        "output_dir": str(final_output_path),
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=to_jsonable)

    print(f"Metrics written to {metrics_path}")
    print(f"JSON summary written to {metrics_json_path}")

    if args.no_save_figures:
        print("Figures were not saved.")
    else:
        print(f"Figures saved under {figures_root}")
        print(f"Saved figure samples: {figure_count}")

    print("done")


if __name__ == "__main__":
    main()