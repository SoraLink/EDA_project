from __future__ import annotations
from typing import Tuple, Dict
import numpy as np

try:
    from skimage import color as skcolor

    _has_lab = True
except ImportError:
    _has_lab = False


def _rgb2lab(rgb: np.ndarray) -> np.ndarray:
    if not _has_lab:
        raise RuntimeError("Lab colours requested but scikit‑image is not installed.")
    lab = skcolor.rgb2lab(rgb / 255.0)
    L = lab[..., 0] / 100.0
    a = (lab[..., 1] + 128.0) / 255.0
    b = (lab[..., 2] + 128.0) / 255.0
    return np.stack([L, a, b], axis=-1).astype(np.float32)


_VIVID = np.array([
    [255, 64, 64],  # red
    [64, 255, 64],  # green
    [64, 64, 255],  # blue
    [255, 255, 64],  # yellow
    [255, 64, 255],  # magenta
    [64, 255, 255],  # cyan
    [255, 128, 0],  # orange
    [128, 0, 255],  # violet
], dtype=np.uint8)


def _make_palette(K: int, mode: str, rng: np.random.Generator | None = None) -> np.ndarray | None:
    mode = mode.strip().lower()
    if mode == "mean":
        return None
    if mode == "vivid":
        if K > len(_VIVID):
            raise ValueError("vivid palette supports up to %d clusters" % len(_VIVID))
        return _VIVID[:K]
    if mode == "random":
        rng = rng or np.random.default_rng()
        return rng.integers(0, 256, size=(K, 3), dtype=np.uint8)
    raise ValueError(f"Unknown palette mode: {mode}")


def _kmeans_pp_init(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    N, D = X.shape
    centers = np.empty((K, D), dtype=X.dtype)
    centers[0] = X[rng.integers(N)]
    for k in range(1, K):
        dist_sq = np.min(((X - centers[:k, None]) ** 2).sum(axis=2), axis=0)
        centers[k] = X[rng.choice(N, p=dist_sq / dist_sq.sum())]
    return centers


def kmeans(
        X: np.ndarray,
        K: int,
        *,
        n_init: int = 10,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    rng_master = np.random.default_rng(seed)
    best_inertia = np.inf
    best_labels = best_centers = None

    for _ in range(n_init):
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        centers = _kmeans_pp_init(X, K, rng)

        for _ in range(max_iter):
            d2 = ((X - centers[:, None]) ** 2).sum(axis=2)
            labels = np.argmin(d2, axis=0)

            new_centers = np.stack([
                X[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
                for k in range(K)
            ])
            if np.linalg.norm(new_centers - centers) < tol:
                centers = new_centers
                break
            centers = new_centers

        inertia = ((X - centers[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia, best_labels, best_centers = inertia, labels.copy(), centers.copy()

    return best_labels, best_centers, float(best_inertia)


def segment_image(
        img: np.ndarray,
        *,
        K: int = 4,
        colorspace: str = "lab",
        palette: str = "mean",
        **km_kwargs,
) -> Tuple[np.ndarray, Dict[str, float]]:
    H, W, _ = img.shape

    if colorspace.lower() == "lab":
        feats = _rgb2lab(img)
    elif colorspace.lower() == "rgb":
        feats = img.astype(np.float32) / 255.0
    else:
        raise ValueError("colorspace must be 'rgb' or 'lab'")

    X = feats.reshape(-1, 3)
    labels, _, inertia = kmeans(X, K, **km_kwargs)

    palette_arr = _make_palette(K, palette)
    if palette_arr is None:
        palette_arr = np.stack([
            img.reshape(-1, 3)[labels == k].mean(axis=0) if (labels == k).any() else [0, 0, 0]
            for k in range(K)
        ])

    seg = palette_arr[labels].reshape(H, W, 3).astype(np.uint8)
    return seg, {"sse": inertia}


__all__ = [
    "segment_image",
    "kmeans",
]
