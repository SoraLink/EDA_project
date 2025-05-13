from __future__ import annotations
from typing import Final, Tuple
import numpy as np
from numpy.typing import NDArray
from algos.image_segement_algorithm import ImageSegmentAlgorithm

try:
    from skimage import color as skcolor
    _HAS_LAB: Final[bool] = True
except ImportError:
    _HAS_LAB = False


def _rgb2lab(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """
    Convert an 8-bit RGB image to normalized Lab space.

    Parameters
    ----------
    rgb : ndarray of uint8, shape (H, W, 3)
        Input image in [0,255] RGB.

    Returns
    -------
    lab_norm : ndarray of float32, shape (H, W, 3)
        Lab channels normalized to [0,1].
    """

    if not _HAS_LAB:
        raise RuntimeError("scikit-image is required for Lab colour space.")
    lab = skcolor.rgb2lab(rgb / 255.0)
    L = lab[..., 0] / 100.0
    a = (lab[..., 1] + 128.0) / 255.0
    b = (lab[..., 2] + 128.0) / 255.0
    return np.stack([L, a, b], axis=-1).astype(np.float32)

# Predefined vivid colors palette
_VIVID: NDArray[np.uint8] = np.array(
    [
        [255, 64, 64],
        [64, 255, 64],
        [64, 64, 255],
        [255, 255, 64],
        [255, 64, 255],
        [64, 255, 255],
        [255, 128, 0],
        [128, 0, 255],
    ],
    dtype=np.uint8,
)


def _make_palette(K: int,
                  mode: str,
                  rng: np.random.Generator | None = None) -> NDArray[np.uint8] | None:
    """
    Generate or select a color palette for visualizing K clusters.

    Parameters
    ----------
    K : int
        Number of clusters.
    mode : str
        Palette mode: 'mean', 'vivid', or 'random'.
    rng : Generator, optional
        Random number generator for 'random' mode.

    Returns
    -------
    palette : ndarray of uint8, shape (K,3), or None
        RGB colors for each cluster, or None to use mean colors.
    """
    mode = mode.strip().lower()
    if mode == "mean":
        return None
    if mode == "vivid":
        if K <= len(_VIVID):
            return _VIVID[:K]
        print(f"[Warning] 'vivid' offers only {len(_VIVID)} colours; switching to random.")
        mode = "random"
    if mode == "random":
        rng = rng or np.random.default_rng()
        return rng.integers(0, 256, size=(K, 3), dtype=np.uint8)
    raise ValueError(f"Unknown palette mode: {mode}")


def _pairwise_sq_dists(C: NDArray[np.float32],
                       X: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Compute pairwise squared Euclidean distances between centres and points.

    Parameters
    ----------
    C : ndarray of shape (M, D)
        Cluster centres.
    X : ndarray of shape (N, D)
        Data points.

    Returns
    -------
    d2 : ndarray of shape (M, N)
        Squared distances.
    """
    x2 = (X ** 2).sum(axis=1)
    c2 = (C ** 2).sum(axis=1)
    d2 = c2[:, None] + x2[None, :] - 2.0 * C @ X.T
    return np.clip(d2, 0.0, None)


def _kmeans_pp_init(X: NDArray[np.float32],
                    K: int,
                    rng: np.random.Generator) -> NDArray[np.float32]:
    """
    Initialize K-means centres using the k-means++ algorithm.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Data points.
    K : int
        Number of clusters.
    rng : Generator
        Random generator for reproducibility.

    Returns
    -------
    centres : ndarray, shape (K, D)
        Initialized cluster centres.
    """

    N, D = X.shape
    centres = np.empty((K, D), dtype=X.dtype)
    centres[0] = X[rng.integers(N)]
    for k in range(1, K):
        dist_sq = np.min(_pairwise_sq_dists(centres[:k], X), axis=0)
        total = dist_sq.sum()
        if total == 0 or not np.isfinite(total):
            centres[k] = X[rng.integers(N)]
        else:
            centres[k] = X[rng.choice(N, p=dist_sq / total)]
    return centres


def _kmeans(X: NDArray[np.float32],
            K: int,
            n_init: int = 10,
            max_iter: int = 100,
            tol_rel: float = 1e-3,
            seed: int | None = 42
            ) -> Tuple[NDArray[np.int32], NDArray[np.float32], float]:
    """
    Run K-means clustering with multiple initializations.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Data points.
    K : int
        Number of clusters.
    n_init : int
        Number of re-starts for better convergence.
    max_iter : int
        Maximum iterations per run.
    tol_rel : float
        Relative centre shift tolerance for convergence.
    seed : int
        Base random seed.

    Returns
    -------
    labels : ndarray of int32, shape (N,)
        Cluster assignment for each point.
    centres : ndarray, shape (K, D)
        Final cluster centres.
    inertia : float
        Sum of squared distances (objective value).
    """

    rng_master = np.random.default_rng(seed)
    best_inertia = np.inf
    best_labels = best_centres = None

    # Multiple restarts
    for _ in range(n_init):
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        centres = _kmeans_pp_init(X, K, rng)
        for _ in range(max_iter):
            d2 = _pairwise_sq_dists(centres.astype(np.float32), X)
            labels = np.argmin(d2, axis=0).astype(np.int32)
            new_centres = centres.copy()
            for k in range(K):
                mask = labels == k
                if mask.any():
                    new_centres[k] = X[mask].mean(axis=0)
                else:
                    new_centres[k] = X[np.argmax(d2[k])]
            delta = np.linalg.norm(new_centres - centres) / (np.linalg.norm(centres) + 1e-9)
            centres = new_centres
            if delta < tol_rel:
                break

        # Compute inertia
        inertia = ((X - centres[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia, best_labels, best_centres = inertia, labels.copy(), centres.copy()
    assert best_labels is not None and best_centres is not None
    return best_labels, best_centres, float(best_inertia)


def _segment_image(img: NDArray[np.uint8],
                   K: int = 4,
                   colorspace: str = "lab",
                   palette: str = "mean",
                   **km_kwargs):
    """
    Segment an image via K-means on pixel colour features.

    Parameters
    ----------
    img : ndarray of uint8, shape (H, W, 3)
        Input RGB image.
    K : int
        Number of segments (must be 4.)
    colorspace : str
        'lab' or 'rgb' feature space.
    palette : str
        Colour mode for visualization.
    km_kwargs : dict
        Additional parameters for K-means.

    Returns
    -------
    seg_vis : ndarray of uint8, shape (H, W, 3)
        Colour segmentation map.
    info : dict
        Contains 'labels' ndarray(H,W) and 'sse' inertia.
    """
    if K != 4:
        raise ValueError("This project requires K == 4.")
    H, W, _ = img.shape

    # Choose feature space
    if colorspace.lower() == "lab":
        feats = _rgb2lab(img)
    elif colorspace.lower() == "rgb":
        feats = img.astype(np.float32) / 255.0
    else:
        raise ValueError("colorspace must be 'rgb' or 'lab'.")
    X = feats.reshape(-1, 3)
    labels, _, inertia = _kmeans(X, K, **km_kwargs)
    palette_arr = _make_palette(K, palette)
    if palette_arr is None:

        # Mean colour of each cluster
        palette_arr = np.stack([
            img.reshape(-1, 3)[labels == k].mean(axis=0) if (labels == k).any() else [0, 0, 0]
            for k in range(K)
        ]).astype(np.uint8)

    # Create visualization image
    seg_vis = palette_arr[labels].reshape(H, W, 3).astype(np.uint8)
    return seg_vis, {"sse": inertia, "labels": labels.reshape(H, W)}


class KMeansSegmentation(ImageSegmentAlgorithm):
    """
    Segment images by K-means clustering on RGB or Lab colours.
    """

    def __init__(self,
                 K: int = 4,
                 colorspace: str = "lab",
                 palette: str = "mean",
                 **km_kwargs):
        """
        Initialize segmentation parameters.

        Parameters
        ----------
        K : int
            Number of clusters.
        colorspace : str
            Feature colour space: 'lab' or 'rgb'.
        palette : str
            Visualization palette mode.
        km_kwargs : dict
            Passed to internal K-means.
        """
        self.K = K
        self.colorspace = colorspace
        self.palette = palette
        self.km_kwargs = km_kwargs

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment image and return label map.

        Parameters
        ----------
        image : ndarray, shape (H, W, 3)
            Input RGB image.

        Returns
        -------
        labels : ndarray of int, shape (H, W)
            Cluster labels per pixel.
        """

        _, info = _segment_image(
            image,
            K=self.K,
            colorspace=self.colorspace,
            palette=self.palette,
            **self.km_kwargs
        )
        return info["labels"]

    def segment_full(self, image: np.ndarray):
        """
        Segment image and return full info dictionary.

        Returns
        -------
        seg_vis : ndarray, shape (H, W, 3)
            Colour segmentation map.
        info : dict
            Contains 'labels' and 'sse'.
        """
        return _segment_image(
            image,
            K=self.K,
            colorspace=self.colorspace,
            palette=self.palette,
            **self.km_kwargs
        )
