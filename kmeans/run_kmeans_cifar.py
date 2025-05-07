#把这个复制到PowerShell里，可以从cifar几个类里随机抽取图片进行分割
#python run_kmeans_cifar.py --dataset_root .\cifar-10-batches-py --classes airplane deer horse --sample_one --palette vivid --save_dir samples --compare

from __future__ import annotations
import argparse
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from kmeans_segmentation import segment_image  # local import

LABEL_MAP = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",
}
CLASS_TO_IDX = {v: k for k, v in LABEL_MAP.items()}


def _concat_horiz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([a, b], axis=1)


def _load_batches(root: Path):
    """Yield (img_array, label_idx) from all data_batch_* files."""
    for bf in sorted(root.glob("data_batch_*")):
        with open(bf, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        imgs = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = d[b"labels"]
        for img, lbl in zip(imgs, labels):
            yield img, lbl


def _process_single(img: np.ndarray, lbl_idx: int, args) -> int:
    cls_name = LABEL_MAP[lbl_idx]
    seg, _ = segment_image(img, K=args.K, colorspace=args.colorspace, palette=args.palette)

    base = (
        f"{cls_name}_sample"
        if args.sample_one
        else f"{cls_name}_{np.random.randint(1e6):06d}"
    )

    out_seg = args.save_dir / f"{base}_seg.png"
    Image.fromarray(seg).save(out_seg)

    if args.compare:
        cmp = _concat_horiz(img, seg)
        out_cmp = args.save_dir / f"{base}_compare.png"
        Image.fromarray(cmp).save(out_cmp)
    print(f"Saved {base}")
    return 1  # count one image

def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="K‑means colour segmentation on CIFAR‑10")
    parser.add_argument("--dataset_root", required=True, type=Path, help="Path to cifar-10-batches-py directory")
    parser.add_argument("--classes", nargs="*", default=list(CLASS_TO_IDX.keys()), help="Classes to include (names)")
    parser.add_argument("--sample_one", action="store_true", help="Randomly pick one image per class")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--colorspace", choices=["rgb", "lab"], default="lab")
    parser.add_argument("--palette", choices=["mean", "vivid", "random"], default="mean")
    parser.add_argument("--save_dir", type=Path, default=Path("results"))
    parser.add_argument("--compare", action="store_true", help="Save side‑by‑side comparison")

    args = parser.parse_args(argv)

    classes_idx = {CLASS_TO_IDX[c] for c in args.classes if c in CLASS_TO_IDX}
    if not classes_idx:
        raise SystemExit("[Error] No valid classes specified.")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()

    total_saved = 0
    t0 = time.perf_counter()

    if args.sample_one:
        buffer: dict[int, List[Tuple[int, np.ndarray]]] = {idx: [] for idx in classes_idx}
        for idx, (img, lbl) in enumerate(_load_batches(args.dataset_root)):
            if lbl in buffer:
                buffer[lbl].append((idx, img))
        for lbl, lst in buffer.items():
            if not lst:
                print(f"[Warn] No images found for class {LABEL_MAP[lbl]}")
                continue
            rand_idx = rng.integers(len(lst))
            _, img = lst[rand_idx]
            total_saved += _process_single(img, lbl, args)
    else:
        for img, lbl in _load_batches(args.dataset_root):
            if lbl in classes_idx:
                total_saved += _process_single(img, lbl, args)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal {total_saved} images saved to {args.save_dir} in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()

