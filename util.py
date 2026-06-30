"""
util.py - Dataset loader for pre-split DULRTC_triple dataset.

Expected directory:

DULRTC_triple/
├── BULDING/                     # or BUILDING/
├── STATION/
├── Training_Samples/
│   └── DATA/
├── Validation_Samples/
│   └── DATA/
├── Test_Samples/
│   └── DATA/
└── mask_and_fiber/
    ├── FIBER_1/
    ├── FIBER_5/
    ├── FIBER_10/
    ├── FIBER_15/
    ├── MASK_1/
    ├── MASK_5/
    ├── MASK_10/
    └── MASK_15/

Each sample returns:
    D       : [K, H, W]    full ground-truth radio map in [0, 1]
    Omega   : [K, H, W]    binary observation mask
    B       : [1, H, W]    building map in [0, 1]
    tx_pos  : [R, 2]       grid-normalised TX coords, (x, y) in [-1, 1]
    meta    : dict

Important:
    train.py/test.py should use D_obs = D * Omega before feeding the model.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover
    loadmat = None


# ============================================================================
# Basic helpers
# ============================================================================

def _require_scipy():
    if loadmat is None:
        raise ImportError("This dataset needs scipy. Install it with: pip install scipy")


def _first_present(mat: dict, candidates: list[str], file_path: Path):
    """Return the first existing non-meta key in a .mat file."""
    for key in candidates:
        if key in mat:
            return mat[key]

    valid_keys = [k for k in mat.keys() if not k.startswith("__")]
    if len(valid_keys) == 1:
        return mat[valid_keys[0]]

    raise KeyError(
        f"None of keys {candidates} were found in {file_path}. "
        f"Available keys: {valid_keys}"
    )


def _normalise_minmax(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    amin = float(arr.min())
    amax = float(arr.max())
    if amax - amin < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def _to_khw(arr: np.ndarray, K: Optional[int] = None, name: str = "array") -> np.ndarray:
    """
    Convert common radio-map/mask layouts to [K, H, W].

    Supported:
        [H, W]
        [H, W, K]
        [K, H, W]
    """
    arr = np.asarray(arr)

    # Remove useless singleton dimensions, but keep 2D/3D structure.
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr[None].astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}")

    # [H, W, K]
    if arr.shape[-1] <= 16 and arr.shape[0] > 16 and arr.shape[1] > 16:
        arr = np.transpose(arr, (2, 0, 1))
        return arr.astype(np.float32)

    # [K, H, W]
    if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
        return arr.astype(np.float32)

    # Fallback when K is provided.
    if K is not None:
        if arr.shape[-1] == K:
            return np.transpose(arr, (2, 0, 1)).astype(np.float32)
        if arr.shape[0] == K:
            return arr.astype(np.float32)

    raise ValueError(
        f"Cannot infer [K,H,W] layout for {name}, shape={arr.shape}. "
        f"Expected [H,W,K] or [K,H,W]."
    )


def _to_hw(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """Convert building/station arrays to one [H, W] map when possible."""
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 3:
        # If multi-channel, use max projection as a robust union map.
        if arr.shape[-1] <= 16 and arr.shape[0] > 16 and arr.shape[1] > 16:
            return arr.max(axis=-1).astype(np.float32)
        if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
            return arr.max(axis=0).astype(np.float32)

    raise ValueError(f"Cannot convert {name} with shape {arr.shape} to [H,W].")


def _pixel_to_grid_coords(pix_xy: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Convert pixel coords (x=column, y=row) to grid_sample coords in [-1, 1].
    """
    pix_xy = np.asarray(pix_xy, dtype=np.float32)
    px = pix_xy[..., 0]
    py = pix_xy[..., 1]
    gx = 2.0 * px / max(W - 1, 1) - 1.0
    gy = 2.0 * py / max(H - 1, 1) - 1.0
    return np.stack([gx, gy], axis=-1).astype(np.float32)


def extract_tx_positions(
    tx_map: np.ndarray,
    R: int,
    nms_radius: int = 3,
    min_value: float = 0.05,
) -> np.ndarray:
    """
    Extract R TX pixel positions from a 2D TX heatmap/station map.

    Returns:
        pix_xy: [R, 2], each row is [x, y] in pixel coordinates.
    """
    tx_map = _to_hw(tx_map, name="tx_map")
    H, W = tx_map.shape

    if float(tx_map.max()) <= 0:
        warnings.warn("TX map is all zeros; padding TX positions with image center.")
        center = np.array([[W / 2.0, H / 2.0]], dtype=np.float32)
        return np.repeat(center, R, axis=0)

    tmap = tx_map.astype(np.float32)
    t = torch.from_numpy(tmap)[None, None]
    k = 2 * nms_radius + 1
    pooled = F.max_pool2d(t, kernel_size=k, stride=1, padding=nms_radius)
    peak_mask = (t == pooled) & (t >= min_value * tmap.max())
    peak_vals = (t * peak_mask).reshape(-1)

    n_found = int(peak_mask.sum().item())
    top_k = min(R, max(n_found, 1))
    _, top_idx = peak_vals.topk(top_k)

    rows = (top_idx // W).cpu().numpy()
    cols = (top_idx % W).cpu().numpy()
    found = np.stack([cols, rows], axis=-1).astype(np.float32)

    if top_k < R:
        warnings.warn(f"Only found {top_k} TX peaks, but R={R}; padding with center.")
        center = np.array([[W / 2.0, H / 2.0]], dtype=np.float32)
        pad = np.repeat(center, R - top_k, axis=0)
        found = np.concatenate([found, pad], axis=0)

    return found[:R]


def extract_tx_positions_from_any(tx_arr: np.ndarray, R: int) -> np.ndarray:
    """
    Robust TX extraction.

    Cases:
        [H,W]      : NMS top-R peaks
        [H,W,R]    : one max point per channel
        [R,H,W]    : one max point per channel
    """
    arr = np.asarray(tx_arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return extract_tx_positions(arr, R=R)

    if arr.ndim == 3:
        # [H,W,R]
        if arr.shape[-1] <= 16 and arr.shape[0] > 16 and arr.shape[1] > 16:
            pix = []
            C = arr.shape[-1]
            for r in range(min(R, C)):
                y, x = np.unravel_index(np.argmax(arr[..., r]), arr[..., r].shape)
                pix.append([x, y])
            if len(pix) < R:
                extra = extract_tx_positions(arr.max(axis=-1), R=R - len(pix))
                pix.extend(extra.tolist())
            return np.asarray(pix[:R], dtype=np.float32)

        # [R,H,W]
        if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
            pix = []
            C = arr.shape[0]
            for r in range(min(R, C)):
                y, x = np.unravel_index(np.argmax(arr[r]), arr[r].shape)
                pix.append([x, y])
            if len(pix) < R:
                extra = extract_tx_positions(arr.max(axis=0), R=R - len(pix))
                pix.extend(extra.tolist())
            return np.asarray(pix[:R], dtype=np.float32)

    raise ValueError(f"Cannot extract TX positions from shape {arr.shape}")


def _broadcast_mask_to_khw(mask: np.ndarray, K: int, H: int, W: int) -> np.ndarray:
    mask = _to_khw(mask, K=K, name="Omega")

    if mask.shape[1:] != (H, W):
        raise ValueError(f"Mask spatial shape {mask.shape[1:]} != data shape {(H, W)}")

    if mask.shape[0] == 1 and K > 1:
        mask = np.broadcast_to(mask, (K, H, W)).copy()
    elif mask.shape[0] != K:
        raise ValueError(f"Mask channel K={mask.shape[0]} != data K={K}")

    return (mask > 0.5).astype(np.float32)


# ============================================================================
# DULRTC_triple dataset
# ============================================================================

class DULRTCTripleDataset(Dataset):
    """
    Loader for your pre-split DULRTC_triple dataset.

    Args:
        root:
            /data/home/hky/dataset/DULRTC_triple

        split:
            "train", "val", or "test".
            Also accepts:
                "training", "Training_Samples"
                "validation", "Validation_Samples"
                "Test_Samples"

        omega_num:
            1, 5, 10, or 15.
            Used to select MASK_*/FIBER_* folder.

        mask_type:
            "mask"  -> use mask_and_fiber/MASK_{omega_num}
            "fiber" -> use mask_and_fiber/FIBER_{omega_num}

        tx_source:
            "station" -> use root/STATION for TX positions
            "fiber"   -> use mask_and_fiber/FIBER_{omega_num} for TX positions
            "auto"    -> prefer STATION, fall back to FIBER

    Returns:
        D, Omega, B, tx_pos, meta
    """

    SPLIT_DIRS = {
        "train": "Training_Samples",
        "training": "Training_Samples",
        "Training_Samples": "Training_Samples",

        "val": "Validation_Samples",
        "valid": "Validation_Samples",
        "validation": "Validation_Samples",
        "Validation_Samples": "Validation_Samples",

        "test": "Test_Samples",
        "testing": "Test_Samples",
        "Test_Samples": "Test_Samples",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        omega_num: int = 15,
        mask_type: str = "mask",
        R: int = 3,
        K: int = 3,
        normalise_data: bool = True,
        tx_source: str = "auto",
        data_key: str = "tensor_radiomap",
        building_key: str = "BP",
        station_key: str = "mask",
        mask_key: str = "mask",
    ):
        _require_scipy()
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.omega_num = int(omega_num)
        self.mask_type = mask_type.lower()
        self.R = int(R)
        self.K = int(K)
        self.normalise_data = bool(normalise_data)
        self.tx_source = tx_source.lower()

        self.data_key = data_key
        self.building_key = building_key
        self.station_key = station_key
        self.mask_key = mask_key

        if self.split not in self.SPLIT_DIRS:
            raise ValueError(
                f"Unknown split={split!r}. Use train/val/test or "
                f"Training_Samples/Validation_Samples/Test_Samples."
            )

        split_dir = self.SPLIT_DIRS[self.split]
        self.data_dir = self.root / split_dir / "DATA"

        # The folder name in your message is BULDING, so support both spellings.
        building_candidates = [self.root / "BULDING", self.root / "BUILDING"]
        self.building_dir = next((p for p in building_candidates if p.exists()), None)

        self.station_dir = self.root / "STATION"

        mask_base = self.root / "mask_and_fiber"
        if self.mask_type == "mask":
            self.mask_dir = mask_base / f"MASK_{self.omega_num}"
        elif self.mask_type == "fiber":
            self.mask_dir = mask_base / f"FIBER_{self.omega_num}"
        else:
            raise ValueError("mask_type must be 'mask' or 'fiber'.")

        self.fiber_dir = mask_base / f"FIBER_{self.omega_num}"

        self._check_dirs()

        self.files = sorted(self.data_dir.glob("*.mat"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .mat files found under {self.data_dir}")

    def _check_dirs(self):
        required = [self.data_dir, self.mask_dir]
        for d in required:
            if not d.exists():
                raise FileNotFoundError(f"Missing directory: {d}")

        if self.building_dir is None:
            raise FileNotFoundError(
                f"Missing building directory. Expected either "
                f"{self.root / 'BULDING'} or {self.root / 'BUILDING'}"
            )

        if self.tx_source in ("station", "auto") and not self.station_dir.exists():
            if self.tx_source == "station":
                raise FileNotFoundError(f"Missing STATION directory: {self.station_dir}")
            warnings.warn(f"STATION not found, will try FIBER for TX positions: {self.fiber_dir}")

        if self.tx_source in ("fiber", "auto") and not self.fiber_dir.exists():
            if self.tx_source == "fiber":
                raise FileNotFoundError(f"Missing FIBER directory: {self.fiber_dir}")

    def __len__(self):
        return len(self.files)

    def _matching_file(self, directory: Path, fname: str) -> Path:
        p = directory / fname
        if p.exists():
            return p

        # fallback: same stem, any .mat suffix/case
        candidates = sorted(directory.glob(Path(fname).stem + ".*"))
        candidates = [c for c in candidates if c.suffix.lower() == ".mat"]
        if candidates:
            return candidates[0]

        raise FileNotFoundError(f"Cannot find matching file for {fname} in {directory}")

    def _load_data(self, path: Path) -> np.ndarray:
        mat = loadmat(str(path))
        arr = _first_present(
            mat,
            [
                self.data_key,
                "tensor_radiomap",
                "radiomap",
                "radio_map",
                "DATA",
                "data",
                "D",
                "map",
            ],
            path,
        )
        arr = _to_khw(arr, K=self.K, name="D")

        if self.K is not None and arr.shape[0] != self.K:
            warnings.warn(f"K mismatch in {path.name}: data K={arr.shape[0]}, expected K={self.K}")

        if self.normalise_data:
            arr = _normalise_minmax(arr)

        return arr.astype(np.float32)

    def _load_building(self, fname: str, H: int, W: int) -> np.ndarray:
        path = self._matching_file(self.building_dir, fname)
        mat = loadmat(str(path))
        arr = _first_present(
            mat,
            [
                self.building_key,
                "BP",
                "building",
                "buildings",
                "BUILDING",
                "BULDING",
                "B",
                "data",
                "mask",
            ],
            path,
        )
        B = _to_hw(arr, name="B")
        if B.shape != (H, W):
            raise ValueError(f"Building shape {B.shape} != data shape {(H, W)} in {path}")
        B = (B > 0.5).astype(np.float32)
        return B[None]  # [1,H,W]

    def _load_mask(self, fname: str, K: int, H: int, W: int) -> np.ndarray:
        path = self._matching_file(self.mask_dir, fname)
        mat = loadmat(str(path))
        arr = _first_present(
            mat,
            [
                self.mask_key,
                "mask",
                "MASK",
                "Omega",
                "OMEGA",
                "omega",
                "fiber",
                "FIBER",
                "data",
            ],
            path,
        )
        return _broadcast_mask_to_khw(arr, K=K, H=H, W=W)

    def _load_tx(self, fname: str, H: int, W: int) -> np.ndarray:
        tx_arr = None
        tx_path = None

        if self.tx_source in ("station", "auto") and self.station_dir.exists():
            try:
                tx_path = self._matching_file(self.station_dir, fname)
                mat = loadmat(str(tx_path))
                tx_arr = _first_present(
                    mat,
                    [
                        self.station_key,
                        "mask",
                        "station",
                        "STATION",
                        "TXPOS",
                        "txpos",
                        "tx",
                        "antenna",
                        "data",
                    ],
                    tx_path,
                )
            except Exception as e:
                if self.tx_source == "station":
                    raise e
                warnings.warn(f"Failed to load TX from STATION for {fname}: {e}")

        if tx_arr is None and self.tx_source in ("fiber", "auto"):
            tx_path = self._matching_file(self.fiber_dir, fname)
            mat = loadmat(str(tx_path))
            tx_arr = _first_present(
                mat,
                [
                    self.station_key,
                    "mask",
                    "fiber",
                    "FIBER",
                    "station",
                    "TXPOS",
                    "txpos",
                    "tx",
                    "data",
                ],
                tx_path,
            )

        pix_xy = extract_tx_positions_from_any(tx_arr, R=self.R)
        grid_xy = _pixel_to_grid_coords(pix_xy, H=H, W=W)
        return grid_xy.astype(np.float32)

    def __getitem__(self, idx: int):
        data_path = self.files[idx]
        fname = data_path.name

        D = self._load_data(data_path)
        K, H, W = D.shape

        B = self._load_building(fname, H=H, W=W)
        Omega = self._load_mask(fname, K=K, H=H, W=W)
        tx_pos = self._load_tx(fname, H=H, W=W)

        meta = {
            "sample_id": fname,
            "split": self.SPLIT_DIRS[self.split],
            "data_path": str(data_path),
            "omega_num": self.omega_num,
            "mask_type": self.mask_type,
        }

        return (
            torch.from_numpy(np.ascontiguousarray(D)).float(),
            torch.from_numpy(np.ascontiguousarray(Omega)).float(),
            torch.from_numpy(np.ascontiguousarray(B)).float(),
            torch.from_numpy(np.ascontiguousarray(tx_pos)).float(),
            meta,
        )


# ============================================================================
# Optional synthetic dataset, kept for smoke tests
# ============================================================================

def gen_sparse_mask(shape, ratio: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random(shape) < ratio).astype(np.float32)


class SyntheticBTDDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 64,
        H: int = 64,
        W: int = 64,
        K: int = 3,
        R: int = 3,
        omega_ratio: float = 0.1,
        noise_std: float = 0.01,
        seed: int = 0,
    ):
        self.n = int(n_samples)
        self.H = int(H)
        self.W = int(W)
        self.K = int(K)
        self.R = int(R)
        self.omega_ratio = float(omega_ratio)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)

        grid_xy = rng.uniform(low=-0.8, high=0.8, size=(self.R, 2)).astype(np.float32)
        B = (rng.random((self.H, self.W)) < 0.1).astype(np.float32)

        ys = np.linspace(-1, 1, self.H, dtype=np.float32)
        xs = np.linspace(-1, 1, self.W, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        S = np.zeros((self.R, self.H, self.W), dtype=np.float32)
        for r in range(self.R):
            d = np.sqrt((xx - grid_xy[r, 0]) ** 2 + (yy - grid_xy[r, 1]) ** 2)
            S[r] = np.exp(-2.0 * d)

        c = rng.uniform(0.3, 1.0, size=(self.R, self.K)).astype(np.float32)
        D = np.einsum("rhw,rk->khw", S, c)
        D = D + rng.normal(scale=self.noise_std, size=D.shape).astype(np.float32)
        D = _normalise_minmax(D)

        Omega = gen_sparse_mask(D.shape, self.omega_ratio, rng)

        return (
            torch.from_numpy(D).float(),
            torch.from_numpy(Omega).float(),
            torch.from_numpy(B[None]).float(),
            torch.from_numpy(grid_xy).float(),
            {"sample_id": f"synth_{idx}"},
        )


# Backward-compatible alias.
# If train.py still imports BARTLabDataset, it can temporarily point to this class.
BARTLabDataset = DULRTCTripleDataset


if __name__ == "__main__":
    root = "/data/home/hky/dataset/DULRTC_triple"
    ds = DULRTCTripleDataset(root=root, split="train", omega_num=15, mask_type="mask", R=3, K=3)
    print("num samples:", len(ds))
    D, Om, B, tx, meta = ds[0]
    print("D     ", tuple(D.shape), D.min().item(), D.max().item())
    print("Omega ", tuple(Om.shape), Om.mean().item())
    print("B     ", tuple(B.shape), B.min().item(), B.max().item())
    print("tx    ", tuple(tx.shape), tx)
    print("meta  ", meta)
