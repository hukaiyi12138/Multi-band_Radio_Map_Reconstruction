"""
train.py - Train PR-BTD-DULRTC with fixed N_iter.

This script trains one PR-BTD-DULRTC model for one mask setting.
Default N_iter is fixed to 10. It is no longer designed for N_iter ablation.

Loss:
    train loss = L1(D_hat, D) + mu_grad * gradient_L1(D_hat, D)
    val   loss = L1(D_hat, D)          # kept pure-L1 so best.pt stays comparable

The gradient term penalizes the difference between predicted and ground-truth
spatial gradients, forcing the model to recover sharp shadow edges behind
buildings (pure L1 alone has no constraint on edge sharpness and over-smooths).

Example output dirs:
    runs/mask1/
    runs/mask5/
    runs/mask10/
    runs/mask15/
    runs/fiber1/
    runs/fiber5/
    runs/fiber10/
    runs/fiber15/

Recommended command:

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 1 \
    --mask-type fiber \
    --R 3 \
    --K 3 \
    --N-iter 10 \
    --n-ray-samples 16 \
    --prox-hidden 32 \
    --prox-layers 3 \
    --epochs 50 \
    --batch-size 1 \
    --lr 1e-3 \
    --mu-grad 0 \
    --save-root runs
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import PR_BTD_DULRTC
from util import DULRTCTripleDataset, SyntheticBTDDataset


# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()

    # ---------------- dataset ----------------
    p.add_argument(
        "--dataset",
        choices=["dulrtc_triple", "synthetic"],
        default="dulrtc_triple",
    )

    p.add_argument(
        "--root",
        type=str,
        default="/data/home/hky/dataset/DULRTC_triple",
        help="Root path of DULRTC_triple dataset.",
    )

    p.add_argument(
        "--omega-num",
        type=int,
        default=15,
        choices=[1, 5, 10, 15],
        help="Use MASK_{omega_num} or FIBER_{omega_num}.",
    )

    p.add_argument(
        "--mask-type",
        choices=["mask", "fiber"],
        default="mask",
        help="Use MASK_* or FIBER_* as sparse observation mask.",
    )

    # ---------------- model ----------------
    p.add_argument("--R", type=int, default=3)
    p.add_argument("--K", type=int, default=3)

    p.add_argument(
        "--N-iter",
        type=int,
        default=10,
        help="Number of unrolled BTD-ADMM iterations. Default: 10.",
    )

    p.add_argument("--prox-hidden", type=int, default=32)
    p.add_argument("--prox-layers", type=int, default=3)
    p.add_argument("--n-ray-samples", type=int, default=16)

    # ---------------- training ----------------
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)

    # ---------------- loss ----------------
    p.add_argument(
        "--mu-grad",
        type=float,
        default=0.15,
        help=(
            "Weight of the gradient (edge-sharpness) loss. "
            "Set 0.0 to recover the original pure-L1 training. "
            "Start at 0.1; raise to 0.3/0.5 if shadow edges stay blurry, "
            "lower if ringing/overshoot artifacts appear near edges."
        ),
    )

    # ---------------- save ----------------
    p.add_argument(
        "--save-root",
        type=str,
        default="runs",
        help="Root directory for saving training results.",
    )

    p.add_argument(
        "--save-prefix",
        type=str,
        default="",
        help=(
            "Optional custom save folder name. "
            "If empty, use '{mask_type}{omega_num}', e.g. mask15 or fiber1."
        ),
    )

    p.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="If set, save epoch1.pt, epoch2.pt, ... for every epoch.",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
def gradient_l1(D_hat: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Edge-sharpness loss: L1 difference between the spatial gradients of the
    prediction and the ground truth.

    Pure pixel-wise L1 has no constraint on relationships between neighbouring
    pixels, so it does not penalize blurred shadow edges. This term compares
    horizontal and vertical finite differences, forcing predicted edges
    (e.g. building-cast shadow boundaries) to match the ground truth.

    D_hat, D : [N, K, H, W]
    """
    dx_hat = D_hat[:, :, :, 1:] - D_hat[:, :, :, :-1]
    dy_hat = D_hat[:, :, 1:, :] - D_hat[:, :, :-1, :]
    dx_gt = D[:, :, :, 1:] - D[:, :, :, :-1]
    dy_gt = D[:, :, 1:, :] - D[:, :, :-1, :]
    return (dx_hat - dx_gt).abs().mean() + (dy_hat - dy_gt).abs().mean()


# ---------------------------------------------------------------------------
def make_dataset(args, split: str):
    """
    split:
        train -> Training_Samples/DATA
        val   -> Validation_Samples/DATA
    """

    if args.dataset == "synthetic":
        return SyntheticBTDDataset(
            n_samples=512,
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
    """
    Dataset returns:
        D, Omega, B, tx_pos, meta

    Training only needs:
        D, Omega, B, tx_pos
    """
    Ds = torch.stack([b[0] for b in batch])
    Oms = torch.stack([b[1] for b in batch])
    Bs = torch.stack([b[2] for b in batch])
    Tx = torch.stack([b[3] for b in batch])

    return Ds, Oms, Bs, Tx


# ---------------------------------------------------------------------------
def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
def build_loaders(args, device):
    train_ds = make_dataset(args, split="train")
    val_ds = make_dataset(args, split="val")

    print(f"[setup] dataset: {args.dataset}")
    print(f"[setup] train samples: {len(train_ds)}")
    print(f"[setup] val samples:   {len(val_ds)}")
    print(f"[setup] omega: {args.mask_type.upper()}_{args.omega_num}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
def make_run_save_dir(args) -> Path:
    if args.save_prefix:
        exp_name = args.save_prefix
    else:
        prefix = "fiber" if args.mask_type.lower() == "fiber" else "mask"
        exp_name = f"{prefix}{args.omega_num}"

    return Path(args.save_root) / exp_name


# ---------------------------------------------------------------------------
def save_args(save_dir: Path, args):
    args_dict = vars(args).copy()

    # Keep both keys for compatibility with older test.py.
    args_dict["N_iter"] = int(args.N_iter)
    args_dict["N_iters"] = [int(args.N_iter)]

    with open(save_dir / "args.json", "w") as f:
        json.dump(args_dict, f, indent=2)


# ---------------------------------------------------------------------------
def train_one_run(args, train_loader, val_loader, device):
    save_dir = make_run_save_dir(args)
    args.save_dir = str(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_args(save_dir, args)

    print("\n" + "=" * 80)
    print(f"[run] Start training")
    print(f"[run] mask: {args.mask_type}{args.omega_num}")
    print(f"[run] N_iter: {args.N_iter}")
    print(f"[run] mu_grad: {args.mu_grad}")
    print(f"[run] save_dir: {save_dir}")
    print("=" * 80)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = PR_BTD_DULRTC(
        R=args.R,
        K=args.K,
        N_iter=args.N_iter,
        prox_hidden=args.prox_hidden,
        prox_layers=args.prox_layers,
        n_ray_samples=args.n_ray_samples,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] model parameters: {n_params:,}")
    print(f"[setup] R={args.R}, K={args.K}, N_iter={args.N_iter}")
    print(f"[setup] prox_hidden={args.prox_hidden}, n_ray_samples={args.n_ray_samples}")

    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=args.epochs * max(1, len(train_loader)),
    )

    loss_fn = torch.nn.L1Loss()

    log_path = save_dir / "log.txt"
    log_path.touch()

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()

        t0 = time.time()
        sum_loss = 0.0       # total (L1 + grad) for logging
        sum_l1 = 0.0         # L1 part only, for logging
        sum_grad = 0.0       # grad part only, for logging
        n_seen = 0
        n_skip = 0

        for step, (D, Om, B, Tx) in enumerate(train_loader):
            D = D.to(device, non_blocking=True)
            Om = Om.to(device, non_blocking=True)
            B = B.to(device, non_blocking=True)
            Tx = Tx.to(device, non_blocking=True)

            D_obs = D * Om

            opt.zero_grad(set_to_none=True)

            try:
                D_hat, X, E, S, c = model(D_obs, Om, B, Tx)

                l1_term = loss_fn(D_hat, D)
                if args.mu_grad > 0.0:
                    grad_term = gradient_l1(D_hat, D)
                    loss = l1_term + args.mu_grad * grad_term
                else:
                    grad_term = torch.zeros((), device=D.device)
                    loss = l1_term

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"  [warn] CUDA OOM, epoch {epoch + 1}, step {step}; "
                        f"skipping this batch."
                    )
                    opt.zero_grad(set_to_none=True)
                    clear_cuda_cache()
                    n_skip += 1
                    continue
                raise e

            except Exception as e:
                print(
                    f"  [warn] forward failed, epoch {epoch + 1}, step {step}: {e}"
                )
                opt.zero_grad(set_to_none=True)
                clear_cuda_cache()
                n_skip += 1
                continue

            if not torch.isfinite(loss):
                print(
                    f"  [warn] non-finite loss, epoch {epoch + 1}, step {step}; "
                    f"skipping."
                )
                opt.zero_grad(set_to_none=True)
                clear_cuda_cache()
                n_skip += 1
                continue

            try:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

                bad_grad = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_grad = True
                        break

                if bad_grad:
                    print(
                        f"  [warn] non-finite grad, epoch {epoch + 1}, step {step}; "
                        f"skipping update."
                    )
                    opt.zero_grad(set_to_none=True)
                    clear_cuda_cache()
                    n_skip += 1
                    continue

                opt.step()
                sched.step()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"  [warn] CUDA OOM during backward, "
                        f"epoch {epoch + 1}, step {step}; skipping batch."
                    )
                    opt.zero_grad(set_to_none=True)
                    clear_cuda_cache()
                    n_skip += 1
                    continue
                raise e

            bs = D.shape[0]
            sum_loss += loss.item() * bs
            sum_l1 += l1_term.item() * bs
            sum_grad += float(grad_term) * bs
            n_seen += bs

            del D_obs, D_hat, X, E, S, c, loss, l1_term, grad_term
            clear_cuda_cache()

        train_total = sum_loss / max(1, n_seen)
        train_l1 = sum_l1 / max(1, n_seen)
        train_grad = sum_grad / max(1, n_seen)

        # ------------------------------------------------- validation
        # Validation uses pure L1 only, so best.pt selection stays comparable
        # with earlier (pure-L1) runs.
        model.eval()

        sum_val = 0.0
        n_val_seen = 0
        n_val_skip = 0

        with torch.no_grad():
            for step, (D, Om, B, Tx) in enumerate(val_loader):
                D = D.to(device, non_blocking=True)
                Om = Om.to(device, non_blocking=True)
                B = B.to(device, non_blocking=True)
                Tx = Tx.to(device, non_blocking=True)

                D_obs = D * Om

                try:
                    D_hat, X, E, S, c = model(D_obs, Om, B, Tx)
                    val_loss = loss_fn(D_hat, D).item()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(
                            f"  [warn] CUDA OOM during validation, step {step}; "
                            f"skipping."
                        )
                        clear_cuda_cache()
                        n_val_skip += 1
                        continue
                    raise e

                sum_val += val_loss * D.shape[0]
                n_val_seen += D.shape[0]

                del D_obs, D_hat, X, E, S, c
                clear_cuda_cache()

        val_l1 = sum_val / max(1, n_val_seen)
        dt = time.time() - t0

        line = (
            f"[epoch {epoch + 1:3d}/{args.epochs:3d}] "
            f"N_iter {args.N_iter:2d}   "
            f"train total {train_total:.4f}   "
            f"train L1 {train_l1:.4f}   "
            f"train grad {train_grad:.4f}   "
            f"val L1 {val_l1:.4f}   "
            f"lr {opt.param_groups[0]['lr']:.2e}   "
            f"train skips {n_skip}   "
            f"val skips {n_val_skip}   "
            f"{dt:.1f}s"
        )

        print(line)

        with open(log_path, "a") as f:
            f.write(line + "\n")

        ck = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch,
            "epoch_1based": epoch + 1,
            "N_iter": int(args.N_iter),
            "best_val": best_val,
            "best_epoch": best_epoch,
            "args": vars(args),
        }

        if args.save_every_epoch:
            epoch_ckpt_path = save_dir / f"epoch{epoch + 1}.pt"
            torch.save(ck, epoch_ckpt_path)
            print(f"  [save] epoch checkpoint: {epoch_ckpt_path.name}")

        if val_l1 < best_val:
            best_val = val_l1
            best_epoch = epoch + 1
            ck["best_val"] = best_val
            ck["best_epoch"] = best_epoch
            torch.save(ck, save_dir / "best.pt")
            print(f"  [save] new best val L1 = {best_val:.4f} at epoch {best_epoch}")

        clear_cuda_cache()

    summary = {
        "N_iter": int(args.N_iter),
        "best_val": best_val,
        "best_epoch": best_epoch,
        "save_dir": str(save_dir),
        "mask_type": args.mask_type,
        "omega_num": args.omega_num,
        "mu_grad": args.mu_grad,
    }

    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[done] {args.mask_type}{args.omega_num}, "
        f"N_iter={args.N_iter}, best val L1={best_val:.4f}, "
        f"best epoch={best_epoch}, save_dir={save_dir}"
    )

    del model, opt, sched
    clear_cuda_cache()

    return summary


# ---------------------------------------------------------------------------
def main():
    args = get_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"[setup] device: {device}")

    if device.type == "cuda":
        print(f"[setup] cuda device count visible to this process: {torch.cuda.device_count()}")
        print(f"[setup] current cuda device: {torch.cuda.current_device()}")
        print(f"[setup] cuda name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    print(f"[setup] fixed N_iter: {args.N_iter}")
    print(f"[setup] mu_grad: {args.mu_grad}")
    print(f"[setup] save_root: {args.save_root}")

    train_loader, val_loader = build_loaders(args, device)

    summary = train_one_run(
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    print("\n" + "=" * 80)
    print("[training done] Summary")
    print(
        f"{summary['mask_type']}{summary['omega_num']} | "
        f"N_iter={summary['N_iter']} | "
        f"best val L1={summary['best_val']:.4f} | "
        f"best epoch={summary['best_epoch']:3d} | "
        f"{summary['save_dir']}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()