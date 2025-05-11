import argparse
import os
import time

import cv2
import numpy as np

from algos import GSC
from algos.GSC import SuperpixelMethod, ClusteringMethod, GraphBasedSuperpixel
from algos.image_segement_algorithm import ImageSegmentAlgorithm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default="./data/image"
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default="./data/masks"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output"
    )
    subparsers = parser.add_subparsers(
        title='algo',
        dest='algo',
        required=True,
    )
    spectral = subparsers.add_parser(
        "spectral"
    )
    spectral.add_argument(
        "--sigma_I",
        type=float,
        help="sigma_I",
        default=0.05
    )
    spectral.add_argument(
        "--sigma_X",
        type=float,
        help="sigma_X",
        default=4
    )
    spectral.add_argument(
        "--r",
        type=int,
        default=5
    )
    spectral.add_argument(
        "--K",
        type=int,
        default=4
    )
    spectral.add_argument(
        "--target"
    )

    gsc = subparsers.add_parser(
        "gsc"
    )
    gsc.add_argument(
        "--super_pixel_method",
        type=str,
        choices=["SLIC", "LSC", "SEEDS"],
        default="SEEDS"
    )
    gsc.add_argument(
        "--clustering_method",
        type=str,
        choices=["CUT_NORMALIZED", "MERGE_HIERARCHICAL"],
        default="MERGE_HIERARCHICAL"
    )
    gsc.add_argument(
        "--region_size",
        type=int,
        default=20
    )
    gsc.add_argument(
        "--ruler",
        type=float,
        default=10.0
    )
    gsc.add_argument(
        "--ratio",
        type=float,
        default=0.075
    ),
    gsc.add_argument(
        "--num_superpixels",
        type=int,
        default=200
    )
    gsc.add_argument(
        "--num_level",
        type=int,
        default=4
    )
    gsc.add_argument(
        "--prior",
        type=int,
        default=2
    )
    gsc.add_argument(
        "--histogram_bins",
        type=int,
        default=5
    )
    gsc.add_argument(
        "--num_iterations",
        type=int,
        default=10
    )
    gsc.add_argument(
        "--alpha",
        type=float,
        default=0.77
    )
    gsc.add_argument(
        "--beta",
        type=float,
        default=0.03
    )
    gsc.add_argument(
        "--gamma",
        type=float,
        default=0.12
    )
    gsc.add_argument(
        "--delta",
        type=float,
        default=0.1
    )
    gsc.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )
    gsc.add_argument(
        "--K",
        type=int,
        default=2
    )
    return parser

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    A = a.astype(bool)
    B = b.astype(bool)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    if union == 0:
        return 0.0
    return inter / union

def calculate_iou(prediction, label, gt_class: int=1):
    gt_mask = (label == gt_class).astype(np.uint8)

    best_iou = 0.0
    best_class = None
    for c in np.unique(prediction):
        pred_mask = (prediction == c).astype(np.uint8)
        iou = mask_iou(gt_mask, pred_mask)
        if iou > best_iou:
            best_iou = iou
            best_class = c

    return best_class, best_iou

def preprocess_label(label):
    gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
    return mask

def evaluate(image_dir: str, label_dir: str, algo: ImageSegmentAlgorithm) -> (list, list):
    time_consumings = {}
    ious = {}
    for image_path in os.listdir(image_dir):
        time_start = time.time()
        img = cv2.imread(os.path.join(image_dir, image_path))
        prediction = algo.segment(img)
        time_end = time.time()
        time_consumings[image_path] = time_end - time_start
        label = cv2.imread(os.path.join(label_dir, image_path))
        label = preprocess_label(label)
        iou = calculate_iou(prediction, label)
        ious[image_path] = iou
    return time_consumings, ious


def main():
    parser = get_parser()
    args = parser.parse_args()
    algo = None
    if args.algo == "gsc":
        superpixel_method = SuperpixelMethod[args.super_pixel_method]
        clustering_method = ClusteringMethod[args.clustering_method]
        algo = GraphBasedSuperpixel(
            super_pixel_method=superpixel_method,
            clustering_method=clustering_method,
            region_size=args.region_size,
            ruler=args.ruler,
            ratio=args.ratio,
            num_superpixels=args.num_superpixels,
            num_levels=args.num_level,
            prior=args.prior,
            histogram_bins=args.histogram_bins,
            num_iterations=args.num_iterations,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            threshold=args.threshold,
            K=args.K
        )
    else:
        raise NotImplementedError(f"algo {args.algo} not implemented")
    time_consumings, ious = evaluate(args.img_dir, args.label_dir, algo)
    print("time_consumings:", time_consumings)
    print("ious:", ious)

if __name__ == '__main__':
    main()
