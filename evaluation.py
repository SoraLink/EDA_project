import argparse
import os
import time

import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from algos import GSC
from algos.GSC import SuperpixelMethod, ClusteringMethod, GraphBasedSuperpixel
from algos.image_segement_algorithm import ImageSegmentAlgorithm
from algos.kmeans_algorithm import KMeansSegmentation
from algos.spectral_clustering import SpectralClustering

matplotlib.use('Agg')

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

    kmeans = subparsers.add_parser(
        "kmeans"
    )
    kmeans.add_argument(
        "--K",
        type=int,
        default=4
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
        default=0.5
    )
    gsc.add_argument(
        "--beta",
        type=float,
        default=0.3
    )
    gsc.add_argument(
        "--gamma",
        type=float,
        default=0.2
    )
    gsc.add_argument(
        "--delta",
        type=float,
        default=0.0
    )
    gsc.add_argument(
        "--threshold",
        type=float,
        default=0.2
    )
    gsc.add_argument(
        "--K",
        type=int,
        default=2
    )
    return parser

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    Parameters
    ----------
    a : np.ndarray
        First binary mask. Should be a 2D array of 0s and 1s (or boolean).
    b : np.ndarray
        Second binary mask. Should be a 2D array of 0s and 1s (or boolean).

    Returns
    -------
    float
        The IoU score between the two masks. Returns 0.0 if the union is zero.
    """
    A = a.astype(bool)
    B = b.astype(bool)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    if union == 0:
        return 0.0
    return inter / union

def calculate_iou(prediction, label, gt_class: int=1):
    """
    Calculate the best Intersection over Union (IoU) score between predicted mask regions and a specific ground truth class.

    Parameters
    ----------
    prediction : np.ndarray
        The predicted segmentation map with integer class labels.
    label : np.ndarray
        The ground truth segmentation map with integer class labels.
    gt_class : int, optional
        The ground truth class to evaluate against (default is 1).

    Returns
    -------
    best_class : int or None
        The predicted class that gives the highest IoU with the ground truth mask for `gt_class`.
    best_iou : float
        The best IoU score found between the predicted regions and the ground truth class.
    """
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

def compute_boundary_fscore(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int = 2) -> float:
    """
    Compute Boundary F-score between a predicted mask and ground truth mask.

    Parameters:
    -----------
    pred_mask : np.ndarray
        Predicted segmentation mask (2D array, integer-labeled).
    gt_mask : np.ndarray
        Ground truth segmentation mask (2D array, integer-labeled).
    tolerance : int
        Pixel tolerance for boundary matching.

    Returns:
    --------
    bf_score : float
        Boundary F1-score between the predicted and ground truth masks.
    """

    # Convert segmentation masks to binary boundary maps
    pred_boundary = find_boundaries(pred_mask, mode='thick')
    gt_boundary = find_boundaries(gt_mask, mode='thick')

    # Dilate boundaries to allow tolerance in matching
    pred_dil = binary_dilation(pred_boundary, iterations=tolerance)
    gt_dil = binary_dilation(gt_boundary, iterations=tolerance)

    # Precision: fraction of predicted boundary that overlaps ground truth
    precision_hits = np.logical_and(pred_boundary, gt_dil).sum()
    precision_total = pred_boundary.sum()
    precision = precision_hits / (precision_total + 1e-8)

    # Recall: fraction of ground truth boundary that overlaps prediction
    recall_hits = np.logical_and(gt_boundary, pred_dil).sum()
    recall_total = gt_boundary.sum()
    recall = recall_hits / (recall_total + 1e-8)

    # F1 Score
    if precision + recall == 0:
        return 0.0
    bf_score = 2 * precision * recall / (precision + recall)

    return bf_score


def preprocess_label(label):
    gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
    return mask

def evaluate(image_dir: str, label_dir: str, algo: ImageSegmentAlgorithm, output) -> (list, list):
    """
    Evaluate a segmentation algorithm on a dataset of images by computing runtime,
    Intersection over Union (IoU), and Boundary F-score for each image.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing input images.
    label_dir : str
        Path to the directory containing ground truth label images.
    algo : ImageSegmentAlgorithm
        An instance of a segmentation algorithm that implements a `segment(image)` method.
    output : str or None
        Path to a directory where segmentation results (visualized with `label2rgb`) are saved.
        If None or empty, no output is saved.

    Returns
    -------
    time_consumings : dict
        Dictionary mapping image filenames to their segmentation runtime in seconds.
    ious : dict
        Dictionary mapping image filenames to their IoU score with the ground truth.
    bf_scores : dict
        Dictionary mapping image filenames to their Boundary F-score with the ground truth.
    """
    time_consumings = {}
    ious = {}
    bf_scores = {}
    for image_path in tqdm(os.listdir(image_dir)):
        time_start = time.time()
        img = cv2.imread(os.path.join(image_dir, image_path))
        prediction = algo.segment(img)
        if output:
            out = label2rgb(prediction, img, kind='avg')
            plt.imshow(out)
            plt.axis('off')
            plt.savefig(os.path.join(output, image_path))
        time_end = time.time()
        time_consumings[image_path] = time_end - time_start
        label = cv2.imread(os.path.join(label_dir, image_path))
        label = preprocess_label(label)
        iou = calculate_iou(prediction, label)
        ious[image_path] = iou
        bf_score = compute_boundary_fscore(prediction, label)
        bf_scores[image_path] = bf_score
    return time_consumings, ious, bf_scores


def main():
    parser = get_parser()
    args = parser.parse_args()
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
    elif args.algo == "spectral":
        algo = SpectralClustering(
            sigma_I=args.sigma_I,
            sigma_X=args.sigma_X,
            r=args.r,
            k=args.K
        )
    elif args.algo == "kmeans":
        algo = KMeansSegmentation(
            K=args.K,
        )
    else:
        raise NotImplementedError(f"algo {args.algo} not implemented")
    os.makedirs(args.output, exist_ok=True)
    time_consumings, ious, bf_scores = evaluate(args.img_dir, args.label_dir, algo, args.output)
    print("time_consumings:", time_consumings)
    print("ious:", ious)
    print("bf_scores", bf_scores)

if __name__ == '__main__':
    main()
