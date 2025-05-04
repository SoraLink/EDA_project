import os
from enum import IntEnum
import cv2
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation
from skimage import graph
from skimage.color import rgb2lab, label2rgb
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.graph import merge_hierarchical, cut_normalized, rag_mean_color

from algos.image_segement_algorithm import ImageSegmentAlgorithm

class SuperpixelMethod(IntEnum):
    SLIC = 1
    LSC = 2
    SEEDS = 3

class ClusteringMethod(IntEnum):
    CUT_NORMALIZED = 1
    MERGE_HIERARCHICAL = 2

class GraphBasedSuperpixel(ImageSegmentAlgorithm):
    def __init__(
        self,
        super_pixel_method: SuperpixelMethod = SuperpixelMethod.SEEDS,
        clustering_method: ClusteringMethod = ClusteringMethod.MERGE_HIERARCHICAL,
        region_size: int = 20,
        ruler: float = 10.0,
        ratio: float = 0.075,
        num_superpixels: int = 200,
        num_levels: int = 4,
        prior: int = 2,
        histogram_bins: int = 5,
        num_iterations: int = 10,
        alpha: float = 0.77,
        beta: float = 0.03,
        gamma: float = 0.12,
        delta: float = 0.1,
        threshold: float = 0.5,
        K: int = 2
    ):
        # superpixel settings
        self.super_pixel_method = super_pixel_method
        self.region_size = region_size
        self.ruler = ruler
        self.ratio = ratio
        self.num_superpixels = num_superpixels
        self.num_levels = num_levels
        self.prior = prior
        self.histogram_bins = histogram_bins
        self.num_iterations = num_iterations
        # clustering method
        self.clustering_method = clustering_method
        # normalized cut weights
        self.K = K
        # merge weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.threshold = threshold

    def _get_superpixels(self, image: np.ndarray) -> np.ndarray:
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        H, W = image.shape[:2]
        if self.super_pixel_method == SuperpixelMethod.SLIC:
            slic = cv2.ximgproc.createSuperpixelSLIC(
                image_lab, algorithm=cv2.ximgproc.SLIC,
                region_size=self.region_size, ruler=self.ruler
            )
            slic.iterate(self.num_iterations)
            return slic.getLabels()
        if self.super_pixel_method == SuperpixelMethod.LSC:
            lsc = cv2.ximgproc.createSuperpixelLSC(
                image_lab, region_size=self.region_size, ratio=self.ratio
            )
            lsc.iterate(self.num_iterations)
            return lsc.getLabels()
        if self.super_pixel_method == SuperpixelMethod.SEEDS:
            seeds = cv2.ximgproc.createSuperpixelSEEDS(
                W, H, image.shape[2],
                num_superpixels=self.num_superpixels,
                num_levels=self.num_levels,
                prior=self.prior,
                histogram_bins=self.histogram_bins
            )
            seeds.iterate(image, num_iterations=self.num_iterations)
            return seeds.getLabels()
        raise ValueError(f"Unsupported superpixel method: {self.super_pixel_method}")

    def _build_rag(self, image: np.ndarray, labels: np.ndarray) -> graph.RAG:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_lab = rgb2lab(image_rgb)
        rag = rag_mean_color(image_lab, labels)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        self.edge_strength = sobel(gray)
        plt.imshow(self.edge_strength)
        plt.show()
        for edge in rag.edges:
            n1, n2 = edge
            mask1 = labels == n1
            mask2 = labels == n2
            boundary = np.logical_and(mask1, binary_dilation(mask2)) | np.logical_and(mask2, binary_dilation(mask1))
            if np.any(boundary):
                strength = self.edge_strength[boundary].mean()
            else:
                strength = 0.0
            rag.edges[n1, n2]['edge_strength'] = strength
        rag = self.initialize_rag_with_features(rag, labels, image_rgb)
        weights = []
        for u, v in rag.edges:
            cd, ed, td, sd = self._compute_raw_weight(rag, u, v)
            weights.append((cd, ed, td, sd))
        if weights:
            self._max_cd = max(w[0] for w in weights) or 1.0
            self._max_ed = max(w[1] for w in weights) or 1.0
            self._max_td = max(w[2] for w in weights) or 1.0
            self._max_sd = max(w[3] for w in weights) or 1.0
        self.initialize_rag_with_weight(rag, labels)
        return rag

    def _merge_rag_labels(self, labels: np.ndarray, rag: graph.RAG) -> np.ndarray:
        # compute all edge weights
        if self.clustering_method == ClusteringMethod.CUT_NORMALIZED:
            return cut_normalized(labels, rag, num_cuts=self.K)
        elif self.clustering_method == ClusteringMethod.MERGE_HIERARCHICAL:
            self.labels = labels
            return merge_hierarchical(
                labels, rag, thresh=self.threshold,
                rag_copy=False, in_place_merge=True,
                merge_func=self._merge_region_features,
                weight_func=self._weight_region
            )
        raise ValueError(f"Unsupported clustering method: {self.clustering_method}")

    def segment(self, image: np.ndarray) -> np.ndarray:
        sp_labels = self._get_superpixels(image)
        rag = self._build_rag(image, sp_labels)
        merged = self._merge_rag_labels(sp_labels, rag)
        self.draw_rag_overlay(image, merged, rag)
        return merged

    def _merge_region_features(self, graph, src, dst):
        # merge color, edge, texture, and spatial centroid
        tc_dst, pc_dst = graph.nodes[dst]['total color'], graph.nodes[dst]['pixel count']
        tex_dst = graph.nodes[dst]['texture']
        x_dst = graph.nodes[dst]['x_centroid']
        y_dst = graph.nodes[dst]['y_centroid']
        tc_src, pc_src = graph.nodes[src]['total color'], graph.nodes[src]['pixel count']
        tex_src = graph.nodes[src]['texture']
        x_src = graph.nodes[src]['x_centroid']
        y_src = graph.nodes[src]['y_centroid']
        # update stats
        graph.nodes[dst]['total color'] = tc_dst + tc_src
        graph.nodes[dst]['pixel count'] = pc_dst + pc_src
        graph.nodes[dst]['mean color']  = graph.nodes[dst]['total color']/graph.nodes[dst]['pixel count']
        graph.nodes[dst]['texture']       = (tex_dst*pc_dst + tex_src*pc_src)/(pc_dst+pc_src)
        graph.nodes[dst]['x_centroid']    = (x_dst*pc_dst + x_src*pc_src)/(pc_dst+pc_src)
        graph.nodes[dst]['y_centroid']    = (y_dst*pc_dst + y_src*pc_src)/(pc_dst+pc_src)

    def _weight_region(self, graph, src, dst, neighbor):
        # normalized diff in [0,1] by design
        cd = np.linalg.norm(graph.nodes[neighbor]['mean color'][1:] - graph.nodes[dst]['mean color'][1:]) / (self._max_cd + 1e-8)
        td = abs(graph.nodes[neighbor]['texture'] - graph.nodes[dst]['texture']) / (self._max_td + 1e-8)
        dx = graph.nodes[neighbor]['x_centroid'] - graph.nodes[dst]['x_centroid']
        dy = graph.nodes[neighbor]['y_centroid'] - graph.nodes[dst]['y_centroid']
        sd = np.hypot(dx, dy) / (self._max_sd + 1e-8)
        merged_mask = (self.labels == dst) | (self.labels == src)
        mask2 = self.labels == neighbor
        boundary = binary_dilation(merged_mask, np.ones((3, 3))) & mask2
        strength = self.edge_strength[boundary].max() if np.any(boundary) else 0.0
        ed = strength / (self._max_ed + 1e-8)
        w = self.alpha*cd + self.beta * ed + self.gamma*td + self.delta*sd
        return {'weight': w}

    def _compute_raw_weight(self, graph, src, dst):
        cd = np.linalg.norm(graph.nodes[src]['mean color'][1:] - graph.nodes[dst]['mean color'][1:])
        ed = abs(graph.edges[src, dst].get('edge_strength', 0))
        td = abs(graph.nodes[src]['texture'] - graph.nodes[dst]['texture'])
        dx = graph.nodes[src]['x_centroid'] - graph.nodes[dst]['x_centroid']
        dy = graph.nodes[src]['y_centroid'] - graph.nodes[dst]['y_centroid']
        sd = np.hypot(dx, dy)
        return cd, ed, td, sd

    @staticmethod
    def initialize_rag_with_features(rag, labels, image_rgb):
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        # normalized_image = GraphBasedSuperpixel.normalize_lab(image_lab)
        normalized_image = image_rgb.astype(np.float32) / 255.0
        H, W = labels.shape
        img_gray = cv2.cvtColor(cv2.cvtColor(image_rgb,cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2GRAY)
        texture_map = local_binary_pattern(img_gray,P=8,R=1,method='uniform')
        # compute features and centroid
        for region in np.unique(labels):
            mask = labels==region
            pixels = normalized_image[mask]
            xs, ys = np.nonzero(mask)
            rag.nodes[region]['mean color']    = pixels.mean(axis=0)
            rag.nodes[region]['total color']   = pixels.sum(axis=0)
            rag.nodes[region]['pixel count']   = mask.sum()
            rag.nodes[region]['texture']       = texture_map[mask].mean()
            rag.nodes[region]['x_centroid']    = xs.mean()/H
            rag.nodes[region]['y_centroid']    = ys.mean()/W
        return rag

    def initialize_rag_with_weight(self, graph, labels):
        # normalized diff in [0,1] by design
        for u, v in graph.edges:
            cd = np.linalg.norm(graph.nodes[u]['mean color'][1:] - graph.nodes[v]['mean color'][1:]) / (self._max_cd + 1e-8)
            td = abs(graph.nodes[u]['texture'] - graph.nodes[v]['texture']) / (self._max_td + 1e-8)
            dx = graph.nodes[u]['x_centroid'] - graph.nodes[v]['x_centroid']
            dy = graph.nodes[u]['y_centroid'] - graph.nodes[v]['y_centroid']
            sd = np.hypot(dx, dy) / (self._max_sd + 1e-8)
            merged_mask = (labels == u)
            mask2 = labels == v
            boundary = binary_dilation(merged_mask, np.ones((3, 3))) & mask2
            strength = self.edge_strength[boundary].max() if np.any(boundary) else 0.0
            ed = strength / (self._max_ed + 1e-8)
            w = self.alpha*cd + self.beta * ed + self.gamma*td + self.delta*sd
            graph.edges[u, v]['weight'] = w

    @staticmethod
    def normalize_lab(image_lab: np.ndarray) -> np.ndarray:
        """
        Normalize an OpenCV LAB image to [0, 1] range per channel.

        Parameters:
            image_lab (np.ndarray): LAB image in OpenCV format, dtype uint8 or float32.

        Returns:
            np.ndarray: Normalized LAB image in float32, shape same as input.
        """
        image_lab = image_lab.astype(np.float32)
        L = image_lab[:, :, 0] / 100.0  # L channel in [0, 100]
        a = (image_lab[:, :, 1] + 128.0) / 255.0  # a in [-128, 127]
        b = (image_lab[:, :, 2] + 128.0) / 255.0  # b in [-128, 127]
        return np.stack([L, a, b], axis=-1)

    @staticmethod
    def draw_rag_overlay(image, labels, rag, weight_key='weight'):
        # 设置图像背景
        out = label2rgb(labels, image, kind='avg')

        # 创建 NetworkX 图
        G = nx.Graph()
        for n in rag.nodes:
            G.add_node(n)

        # 添加边，边的粗细按 weight 设定
        for u, v, data in rag.edges(data=True):
            weight = data.get(weight_key, 0.0)
            G.add_edge(u, v, weight=weight)

        # 获取每个 superpixel 的质心
        pos = {}
        for region in rag.nodes:
            x = rag.nodes[region]['x_centroid'] * labels.shape[1]
            y = rag.nodes[region]['y_centroid'] * labels.shape[0]
            pos[region] = (y, x)

        # 画图
        plt.figure(figsize=(10, 10))
        plt.imshow(out)

        # 节点位置、大小、边
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color='cyan', alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

        edges = G.edges(data=True)
        weights = [d[2].get(weight_key, 0.0) for d in edges]

        # Normalize weights to line widths
        max_w = max(weights) if weights else 1
        widths = [2.5 * (1 - w / (max_w + 1e-5)) for w in weights]  # weight 越小线越粗

        nx.draw_networkx_edges(G, pos, edge_color='red', width=widths, alpha=0.6)

        plt.axis('off')
        plt.title("RAG with Edge Weights (inverse thickness)")
        plt.show()
        print("\n--- RAG Adjacency Matrix (Weights) ---")
        regions = sorted(rag.nodes)
        matrix = np.full((len(regions), len(regions)), np.nan)

        for u, v, data in G.edges(data=True):
            i = regions.index(u)
            j = regions.index(v)
            matrix[i, j] = matrix[j, i] = data.get(weight_key, 0.0)
        adj_matrix = nx.to_pandas_adjacency(rag, weight=weight_key, nodelist=sorted(rag.nodes()))
        os.makedirs("output", exist_ok=True)
        csv_path = os.path.join("output", "rag_adjacency_matrix.csv")
        adj_matrix.to_csv(csv_path)