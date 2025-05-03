import dataclasses

import cv2
import numpy as np
from skimage import graph
from skimage.color import rgb2lab
from skimage.graph import merge_hierarchical
from webencodings import labels

from algos.image_segement_algorithm import ImageSegmentAlgorithm

@dataclasses
class SuperpixelMethod:
    SLIC = 1
    LSC = 2
    SEEDS = 3

class GraphBasedSuperpixel(ImageSegmentAlgorithm):
    def __init__(
        self,
        super_pixel_method=SuperpixelMethod.LSC,
        region_size: int = 20,
        ruler: float = 10.0,              # SLIC
        ratio: float = 0.075,             # LSC
        num_superpixels: int = 400,       # SEEDS
        num_levels: int = 4,
        prior: int = 2,
        histogram_bins: int = 5,
        num_iterations: int = 10
    ):
        self.super_pixel_method = super_pixel_method
        self.region_size = region_size
        self.ruler = ruler
        self.ratio = ratio
        self.num_superpixels = num_superpixels
        self.num_levels = num_levels
        self.prior = prior
        self.histogram_bins = histogram_bins
        self.num_iterations = num_iterations

    def _get_superpixels(self, image):
        """
        Generate superpixel segmentation labels for the input image using the specified method.

        Depending on the selected `super_pixel_method`, this method applies one of the supported
        superpixel algorithms (SLIC, LSC, or SEEDS) to compute a label map, where each pixel is
        assigned to a superpixel region.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format as loaded by cv2.imread().

        Returns
        -------
        labels : np.ndarray
            A 2D array of shape (H, W), where each element is an integer representing the superpixel
            label assigned to the corresponding pixel.
        """

        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        H, W = image.shape[:2]

        if self.super_pixel_method == SuperpixelMethod.SLIC:
            slic = cv2.ximgproc.createSuperpixelSLIC(
                image_lab,
                algorithm=cv2.ximgproc.SLIC,
                region_size=self.region_size,
                ruler=self.ruler
            )
            slic.iterate(self.num_iterations)
            labels = slic.getLabels()

        elif self.super_pixel_method == SuperpixelMethod.LSC:
            lsc = cv2.ximgproc.createSuperpixelLSC(
                image_lab,
                region_size=self.region_size,
                ratio=self.ratio
            )
            lsc.iterate(self.num_iterations)
            labels = lsc.getLabels()

        elif self.super_pixel_method == SuperpixelMethod.SEEDS:
            seeds = cv2.ximgproc.createSuperpixelSEEDS(
                W, H, image.shape[2],
                num_superpixels=self.num_superpixels,
                num_levels=self.num_levels,
                prior=self.prior,
                histogram_bins=self.histogram_bins
            )
            seeds.iterate(image, num_iterations=self.num_iterations)
            labels = seeds.getLabels()

        else:
            raise ValueError(f"Unsupported superpixel method: {self.super_pixel_method}")

        return labels

    def _build_rag(self, image: np.ndarray, labels: np.ndarray) -> graph.RAG:
        """
        Construct a Region Adjacency Graph (RAG) from superpixel labels.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format.
        labels : np.ndarray
            2D array of superpixel labels, shape (H, W).

        Returns
        -------
        rag : skimage.future.graph.RAG
            A graph where each node represents a superpixel and each edge
            connects adjacent superpixels with edge weights based on mean color distance.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_lab = rgb2lab(image_rgb)
        rag = graph.rag_mean_color(image_lab, labels, mode='distance')
        return rag

    def _merge_rag_labels(self, image: np.ndarray, labels: np.ndarray, rag: graph.RAG,
                          threshold: float = 20.0) -> np.ndarray:
        """
        Merge superpixels using hierarchical merging on the RAG.

        Parameters
        ----------
        image : np.ndarray
            Original input image in BGR format.
        labels : np.ndarray
            Original superpixel labels (H, W).
        rag : skimage.future.graph.RAG
            Region Adjacency Graph built from the superpixels.
        threshold : float
            Merging threshold; lower value means stricter merging.

        Returns
        -------
        merged_labels : np.ndarray
            A 2D array of shape (H, W), where each pixel is assigned to a merged region.
        """
        merged_labels = merge_hierarchical(
            labels, rag, thresh=threshold, rag_copy=False,
            in_place_merge=True,
            merge_func=self._merge_mean_color,
            weight_func=self._weight_mean_color
        )
        return merged_labels

    def segment(self, image: np.ndarray) -> np.ndarray:
        superpixel_labels = self._get_superpixels(image)
        rag = self._build_rag(image, superpixel_labels)
        labels = self._merge_rag_labels(image, superpixel_labels, rag)
        return labels

    @staticmethod
    def _merge_mean_color(graph, src, dst):
        graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
        graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
        graph.nodes[dst]['mean color'] = (
            graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count']
        )

    @staticmethod
    def _weight_mean_color(graph, src, dst, n):
        diff = graph.nodes[src]['mean color'] - graph.nodes[dst]['mean color']
        return np.linalg.norm(diff)