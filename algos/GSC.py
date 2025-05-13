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
from sklearn.cluster import SpectralClustering

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
        """
        Compute superpixel segmentation for the input image.

        Parameters
        ----------
        image : ndarray of shape (H, W, 3)
            Input image in BGR color space.

        Returns
        -------
        labels : ndarray of shape (H, W)
            Label map where each pixel's value corresponds to its superpixel label.
        """

        # Convert BGR image to LAB color space for perceptual clustering
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Extract image dimensions
        H, W = image.shape[:2]

        # ----- SLIC Superpixels -----
        if self.super_pixel_method == SuperpixelMethod.SLIC:
            slic = cv2.ximgproc.createSuperpixelSLIC(
                image_lab, algorithm=cv2.ximgproc.SLIC,
                region_size=self.region_size, ruler=self.ruler
            )
            slic.iterate(self.num_iterations)
            return slic.getLabels()

        # ----- LSC Superpixels -----
        if self.super_pixel_method == SuperpixelMethod.LSC:
            lsc = cv2.ximgproc.createSuperpixelLSC(
                image_lab, region_size=self.region_size, ratio=self.ratio
            )
            lsc.iterate(self.num_iterations)
            return lsc.getLabels()

        # ----- SEEDS Superpixels -----
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
        """
        Build a Region Adjacency Graph (RAG) for the segmented image and compute edge features.

        Parameters
        ----------
        image : ndarray of shape (H, W, 3)
            Input image in BGR color space.
        labels : ndarray of shape (H, W)
            Superpixel label map where each pixel's value indicates its superpixel ID.

        Returns
        -------
        rag : networkx.Graph (or skimage.future.graph.RAG)
            Region Adjacency Graph where nodes represent superpixels and
            edges carry computed edge strength and additional features.
        """

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_lab = rgb2lab(image_rgb)
        rag = rag_mean_color(image_lab, labels)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        self.edge_strength = sobel(gray)

        # Iterate over each adjacency (edge) in the RAG
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

        # Initialize additional region features (e.g., area, color histograms)
        rag = self.initialize_rag_with_features(rag, labels, image_rgb)
        weights = []
        for u, v in rag.edges:
            cd, ed, td, sd = self._compute_raw_weight(rag, u, v)
            weights.append((cd, ed, td, sd))

        # Determine maximum values for normalization (avoid division by zero)
        if weights:
            self._max_cd = max(w[0] for w in weights) or 1.0
            self._max_ed = max(w[1] for w in weights) or 1.0
            self._max_td = max(w[2] for w in weights) or 1.0
            self._max_sd = max(w[3] for w in weights) or 1.0
        self.initialize_rag_with_weight(rag, labels)
        return rag

    def _merge_rag_labels(self, labels: np.ndarray, rag: graph.RAG) -> np.ndarray:
        """
        Merge regions in a Region Adjacency Graph (RAG) into final segments based on the chosen clustering method.

        Parameters
        ----------
        labels : ndarray of shape (H, W)
            Initial superpixel label map where each pixel's value is its superpixel ID.
        rag : networkx.Graph (or skimage.future.graph.RAG)
            Region Adjacency Graph with precomputed edge weights and features.

        Returns
        -------
        merged_labels : ndarray of shape (H, W)
            Label map after merging regions into clusters or hierarchical segments.
        """

        # --- Normalized Cut Clustering ---
        if self.clustering_method == ClusteringMethod.CUT_NORMALIZED:
            dists = [data['weight'] for _, _, data in rag.edges(data=True)]
            min_d, max_d = min(dists), max(dists)
            sim_rag = rag.copy()
            eps = 1e-8

            # Convert distances to similarities: sim = 1 - (d_norm)
            for u, v, data in sim_rag.edges(data=True):
                d_norm = (data['weight'] - min_d) / (max_d - min_d + eps)
                data['weight'] = 1.0 - d_norm

            regions = sorted(sim_rag.nodes())
            A = nx.to_numpy_array(sim_rag, nodelist=regions, weight='weight')

            # Perform spectral clustering on the precomputed affinity matrix
            sc = SpectralClustering(
                n_clusters=self.K,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=0
            )
            labels_flat = sc.fit_predict(A)
            region2cluster = {region: int(labels_flat[i]) for i, region in enumerate(regions)}

            # Apply the region-to-cluster map to the pixel-wise label image
            final_seg = np.vectorize(region2cluster.get)(labels)

            return final_seg

        # --- Hierarchical Merge Clustering ---
        elif self.clustering_method == ClusteringMethod.MERGE_HIERARCHICAL:
            # Store the initial labels for use in merge callbacks
            self.labels = labels
            # Merge regions hierarchically using threshold and custom merge/weight functions
            return merge_hierarchical(
                labels, rag, thresh=self.threshold,
                rag_copy=False, in_place_merge=True,
                merge_func=self._merge_region_features,
                weight_func=self._weight_region
            )
        raise ValueError(f"Unsupported clustering method: {self.clustering_method}")

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Perform full image segmentation by generating superpixels, building a RAG, and merging regions.

        Parameters
        ----------
        image : ndarray of shape (H, W, 3)
            Input image in BGR color space to be segmented.

        Returns
        -------
        merged : ndarray of shape (H, W)
            Final label map after superpixel generation and region merging.
        """

        # Step 1: Generate initial superpixel labels
        sp_labels = self._get_superpixels(image)

        # Step 2: Build a Region Adjacency Graph (RAG) using superpixel labels
        rag = self._build_rag(image, sp_labels)

        # Step 3: Merge adjacent regions in the RAG according to clustering strategy
        merged = self._merge_rag_labels(sp_labels, rag)

        # (Optional) Visualize segmentation by overlaying RAG on image
        # self.draw_rag_overlay(image, merged, rag)

        return merged

    def _merge_region_features(self, graph, src, dst):
        """
        Merge feature statistics of two adjacent regions in the RAG by accumulating
        color, texture, and spatial centroid information from the source region into the destination.

        Parameters
        ----------
        graph : skimage.future.graph.RAG or networkx.Graph
            Region Adjacency Graph containing node attributes for each region.
        src : int
            Node ID of the source region to merge.
        dst : int
            Node ID of the destination region (target of the merge).

        Notes
        -----
        This function updates the 'dst' node in-place, combining the features of
        'src' into 'dst'. After merging, 'dst' will have updated 'total color',
        'pixel count', 'mean color', 'texture', and centroid coordinates.
        """

        # Retrieve destination region statistics
        tc_dst, pc_dst = graph.nodes[dst]['total color'], graph.nodes[dst]['pixel count']
        tex_dst = graph.nodes[dst]['texture']
        x_dst = graph.nodes[dst]['x_centroid']
        y_dst = graph.nodes[dst]['y_centroid']

        # Retrieve source region statistics
        tc_src, pc_src = graph.nodes[src]['total color'], graph.nodes[src]['pixel count']
        tex_src = graph.nodes[src]['texture']
        x_src = graph.nodes[src]['x_centroid']
        y_src = graph.nodes[src]['y_centroid']

        # Update destination's total color and pixel count
        graph.nodes[dst]['total color'] = tc_dst + tc_src
        graph.nodes[dst]['pixel count'] = pc_dst + pc_src

        # Recompute mean color as color sum divided by total pixels
        graph.nodes[dst]['mean color']  = graph.nodes[dst]['total color']/graph.nodes[dst]['pixel count']

        # Merge texture by weighted average based on pixel counts
        graph.nodes[dst]['texture']       = (tex_dst*pc_dst + tex_src*pc_src)/(pc_dst+pc_src)

        # Compute new centroids by weighted average of coordinates
        graph.nodes[dst]['x_centroid']    = (x_dst*pc_dst + x_src*pc_src)/(pc_dst+pc_src)
        graph.nodes[dst]['y_centroid']    = (y_dst*pc_dst + y_src*pc_src)/(pc_dst+pc_src)

    def _weight_region(self, graph, src, dst, neighbor):
        """
        Compute the normalized edge weight between the merged region (src->dst) and a neighbor region.

        Parameters
        ----------
        graph : skimage.future.graph.RAG or networkx.Graph
            Region Adjacency Graph containing node attributes.
        src : int
            ID of the source region being merged into dst.
        dst : int
            ID of the destination region after merging src.
        neighbor : int
            ID of the neighboring region to compute weight with.

        Returns
        -------
        weight_dict : dict
            Dictionary with key 'weight' and normalized weight value combining
            color, edge, texture, and spatial differences.
        """

        # 1. Color difference (skip L channel): Euclidean norm in AB space, normalized
        cd = np.linalg.norm(graph.nodes[neighbor]['mean color'][1:] - graph.nodes[dst]['mean color'][1:]) / (self._max_cd + 1e-8)

        # 2. Texture difference: absolute difference, normalized
        td = abs(graph.nodes[neighbor]['texture'] - graph.nodes[dst]['texture']) / (self._max_td + 1e-8)

        # 3. Spatial distance: Euclidean distance between centroids, normalized
        dx = graph.nodes[neighbor]['x_centroid'] - graph.nodes[dst]['x_centroid']
        dy = graph.nodes[neighbor]['y_centroid'] - graph.nodes[dst]['y_centroid']
        sd = np.hypot(dx, dy) / (self._max_sd + 1e-8)

        # 4. Edge strength: maximum gradient at the boundary between merged region and neighbor
        # Create mask for merged region: union of src and dst pixels
        merged_mask = (self.labels == dst) | (self.labels == src)
        mask2 = self.labels == neighbor
        boundary = binary_dilation(merged_mask, np.ones((3, 3))) & mask2
        strength = self.edge_strength[boundary].max() if np.any(boundary) else 0.0
        ed = strength / (self._max_ed + 1e-8)

        # 5. Combine all normalized components with weights alpha, beta, gamma, delta
        w = self.alpha*cd + self.beta * ed + self.gamma*td + self.delta*sd
        return {'weight': w}

    def _compute_raw_weight(self, graph, src, dst):
        """
        Compute raw (unnormalized) feature differences between two adjacent regions in the RAG.

        Parameters
        ----------
        graph : skimage.future.graph.RAG or networkx.Graph
            Region Adjacency Graph with node and edge attributes.
        src : int
            ID of the source region.
        dst : int
            ID of the destination (adjacent) region.

        Returns
        -------
        cd : float
            Color difference (Euclidean norm in AB color space) between src and dst.
        ed : float
            Edge strength difference, taken from the RAG edge attribute 'edge_strength'.
        td : float
            Texture difference (absolute difference) between src and dst.
        sd : float
            Spatial distance (Euclidean distance) between centroids of src and dst.
        """

        # 1. Color difference (skip L channel): Euclidean norm in AB color space
        cd = np.linalg.norm(graph.nodes[src]['mean color'][1:] - graph.nodes[dst]['mean color'][1:])

        # 2. Edge strength: absolute 'edge_strength' attribute from the RAG
        ed = abs(graph.edges[src, dst].get('edge_strength', 0))

        # 3. Texture difference: absolute difference of texture features
        td = abs(graph.nodes[src]['texture'] - graph.nodes[dst]['texture'])

        # 4. Spatial distance: Euclidean distance between region centroids
        dx = graph.nodes[src]['x_centroid'] - graph.nodes[dst]['x_centroid']
        dy = graph.nodes[src]['y_centroid'] - graph.nodes[dst]['y_centroid']
        sd = np.hypot(dx, dy)

        return cd, ed, td, sd

    @staticmethod
    def initialize_rag_with_features(rag, labels, image_rgb):
        """
        Initialize RAG nodes with color, texture, pixel count, and centroid features.

        Parameters
        ----------
        rag : skimage.future.graph.RAG or networkx.Graph
            Region Adjacency Graph to attach node attributes.
        labels : ndarray of shape (H, W)
            Superpixel label map where each pixel's value is its region ID.
        image_rgb : ndarray of shape (H, W, 3)
            Input image in RGB color space.

        Returns
        -------
        rag : skimage.future.graph.RAG or networkx.Graph
            RAG with updated node attributes for each region:
            - 'mean color' : average RGB color (normalized to [0,1]).
            - 'total color' : sum of RGB colors over all pixels.
            - 'pixel count' : number of pixels in the region.
            - 'texture' : mean Local Binary Pattern (LBP) value.
            - 'x_centroid', 'y_centroid' : normalized centroid coordinates.
        """

        # Convert RGB image to LAB for potential color normalization (optional)
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        # normalized_image = GraphBasedSuperpixel.normalize_lab(image_lab)

        # Normalize RGB image to [0,1] float32
        normalized_image = image_rgb.astype(np.float32) / 255.0
        H, W = labels.shape

        # Convert RGB back to BGR then to grayscale for texture computation
        img_gray = cv2.cvtColor(cv2.cvtColor(image_rgb,cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2GRAY)

        # Compute Local Binary Pattern texture map
        texture_map = local_binary_pattern(img_gray,P=8,R=1,method='uniform')

        # Iterate through each unique region ID
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
        """
        Compute and assign normalized weights to each edge in the RAG combining color, texture,
        spatial distance, and edge strength components.

        Parameters
        ----------
        graph : skimage.future.graph.RAG or networkx.Graph
            Region Adjacency Graph with precomputed node features and edge 'edge_strength'.
        labels : ndarray of shape (H, W)
            Pixel-wise label map of regions used to compute boundaries.

        Returns
        -------
        graph : skimage.future.graph.RAG or networkx.Graph
            The same RAG with updated edge attribute 'weight' for each adjacency.
        """

        # Iterate over each edge (u, v) in the graph
        for u, v in graph.edges:

            # 1. Color difference in AB channels, normalized by maximum
            cd = np.linalg.norm(graph.nodes[u]['mean color'][1:] - graph.nodes[v]['mean color'][1:]) / (self._max_cd + 1e-8)

            # 2. Texture difference, normalized by maximum
            td = abs(graph.nodes[u]['texture'] - graph.nodes[v]['texture']) / (self._max_td + 1e-8)

            # 3. Spatial distance between centroids, normalized
            dx = graph.nodes[u]['x_centroid'] - graph.nodes[v]['x_centroid']
            dy = graph.nodes[u]['y_centroid'] - graph.nodes[v]['y_centroid']
            sd = np.hypot(dx, dy) / (self._max_sd + 1e-8)

            # 4. Edge strength: maximum gradient along boundary between regions u and v
            merged_mask = (labels == u)
            mask2 = labels == v
            boundary = binary_dilation(merged_mask, np.ones((3, 3))) & mask2
            strength = self.edge_strength[boundary].max() if np.any(boundary) else 0.0
            ed = strength / (self._max_ed + 1e-8)

            # 5. Combine components using weight coefficients
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
        """
        Visualize the Region Adjacency Graph (RAG) overlaid on the segmented image.

        Parameters
        ----------
        image : ndarray of shape (H, W, 3)
            Original image in RGB or BGR color space (used as background).
        labels : ndarray of shape (H, W)
            Label map of superpixels; each value indicates a region ID.
        rag : skimage.future.graph.RAG or networkx.Graph
            Region Adjacency Graph with node centroids and edge weights.
        weight_key : str, optional
            Edge attribute key to use for drawing edge thickness (default is 'weight').

        Notes
        -----
        - The function creates an RGB overlay where each superpixel is colored by its average color.
        - RAG nodes are plotted at region centroids, and edges are drawn with thickness inversely
          proportional to the specified weight (thicker lines for smaller weights).
        - Also prints and saves the adjacency matrix of edge weights to 'output/rag_adjacency_matrix.csv'.
        """

        # Generate background by coloring each region with its average original image color
        out = label2rgb(labels, image, kind='avg')

        # Build a NetworkX graph for visualization
        G = nx.Graph()
        for n in rag.nodes:
            G.add_node(n)

        # Add edges and attach weight attributes
        for u, v, data in rag.edges(data=True):
            weight = data.get(weight_key, 0.0)
            G.add_edge(u, v, weight=weight)

        # Compute centroids in pixel coordinates
        pos = {}
        for region in rag.nodes:
            x = rag.nodes[region]['x_centroid'] * labels.shape[1]
            y = rag.nodes[region]['y_centroid'] * labels.shape[0]
            pos[region] = (y, x)

        plt.figure(figsize=(10, 10))
        plt.imshow(out)

        # Draw superpixel nodes
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color='cyan', alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

        # Prepare edge widths inversely related to weights
        edges = G.edges(data=True)
        weights = [d[2].get(weight_key, 0.0) for d in edges]

        # Normalize weights to line widths
        max_w = max(weights) if weights else 1
        widths = [2.5 * (1 - w / (max_w + 1e-5)) for w in weights]  # weight 越小线越粗

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='red', width=widths, alpha=0.6)

        # Finalize plot
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