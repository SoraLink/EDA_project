import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
import time
from skimage.transform import resize
from skimage.filters import gaussian

class SpectralClustering:
    """
    Implementation of image segmentation using Normalized Cut algorithm
    Reference: Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation.
    """
    
    def __init__(self, sigma_I=0.05, sigma_X=4.0, r=5, k=4, target_size=(64, 64), apply_smoothing=True):
        """
        Initialize parameters for Normalized Cut segmentation.

        Parameters
        ----------
        sigma_I : float
            Scale parameter for intensity (color) Gaussian kernel.
        sigma_X : float
            Scale parameter for spatial (coordinate) Gaussian kernel.
        r : int
            Radius for local neighborhood when building similarity graph.
        k : int
            Number of clusters (segments) to produce.
        target_size : tuple of int
            (height, width) to resize input for faster processing.
        apply_smoothing : bool
            If True, apply Gaussian smoothing before feature extraction.
        """

        self.sigma_I = sigma_I
        self.sigma_X = sigma_X
        self.r = r
        self.k = k
        self.target_size = target_size
        self.apply_smoothing = apply_smoothing
    
    def _preprocess_image(self, image):
        """
        Resize and optionally smooth the image.

        Parameters
        ----------
        image : ndarray, shape (H, W, C) or (H, W)
            Input image in RGB or grayscale.

        Returns
        -------
        processed_image : ndarray, shape (target_H, target_W, 3)
            Float image in [0,1], 3 channels.
        """

        # Make a copy to avoid modifying the original
        processed_image = image.copy()
        
        # Check if image is uint8 and convert to float32 (0-1 range)
        if processed_image.dtype == np.uint8:
            processed_image = processed_image.astype(np.float32) / 255.0
        
        # Ensure image has 3 channels (RGB)
        if len(processed_image.shape) == 2:  # Grayscale
            processed_image = np.stack([processed_image] * 3, axis=2)
        elif processed_image.shape[2] > 3:  # RGBA or other
            processed_image = processed_image[:, :, :3]
            
        # Resize image if target size is provided
        if self.target_size is not None:
            processed_image = resize(processed_image, self.target_size, anti_aliasing=True)
        
        # Apply Gaussian smoothing if requested
        if self.apply_smoothing:
            for i in range(processed_image.shape[2]):
                processed_image[:,:,i] = gaussian(processed_image[:,:,i], sigma=1)
        
        return processed_image
    
    def _extract_features(self, image):
        """
        Build per-pixel feature vectors: color + normalized coordinates.

        Parameters
        ----------
        image : ndarray, shape (H, W, 3)
            Preprocessed float image.

        Returns
        -------
        features : ndarray, shape (H*W, 5)
            Each row is [R, G, B, x_norm, y_norm].
        img_shape : tuple of int
            (H, W) of preprocessed image.
        """

        h, w, c = image.shape
            
        # Creating a Coordinate Grid
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Flatten the features into an NxC matrix, where N is the number of pixels and C is the feature dimension
        features = np.zeros((h*w, c+2), dtype=np.float32)
        features[:, :c] = image.reshape(-1, c)        # Color Features
        features[:, c] = x_coords.flatten() / w       # x coordinate (normalized)
        features[:, c+1] = y_coords.flatten() / h     # y coordinate (normalized)
        
        return features, (h, w)
    
    def _compute_similarity_matrix(self, features, img_shape):
        """
        Build a sparse affinity matrix W based on Gaussian kernels.

        Parameters
        ----------
        features : ndarray, shape (N, 5)
            Pixel features.
        img_shape : tuple of int
            (H, W) preprocessed image size.

        Returns
        -------
        W : scipy.sparse.csr_matrix, shape (N, N)
            Sparse similarity (affinity) matrix.
        """

        print("Calculate the similarity matrix...")
        start_time = time.time()
        
        h, w = img_shape
        n_pixels = h * w
        
        # Create a list of rows, columns, and values for a sparse matrix
        rows = []
        cols = []
        vals = []
        
        # Iterate over each pixel
        for i in range(n_pixels):
            # Get the 2D coordinates of the current pixel
            y_i, x_i = i // w, i % w
            
            # Define the search window (reduce calculations)
            y_min = max(0, y_i - self.r)
            y_max = min(h - 1, y_i + self.r)
            x_min = max(0, x_i - self.r)
            x_max = min(w - 1, x_i + self.r)
            
            # Search neighbors only in a local window
            for y_j in range(y_min, y_max + 1):
                for x_j in range(x_min, x_max + 1):
                    j = y_j * w + x_j
                    
                    # Don't connect itself
                    if i == j:
                        continue
                    
                    # Calculating color and spatial distances
                    color_i = features[i, :3]
                    color_j = features[j, :3]
                    pos_i = features[i, 3:]
                    pos_j = features[j, 3:]
                    
                    color_dist_sq = np.sum((color_i - color_j) ** 2)
                    pos_dist_sq = np.sum((pos_i - pos_j) ** 2)
                    
                    # Spatial distance check: skip points whose distance exceeds r
                    if np.sqrt(pos_dist_sq) > self.r:
                        continue
                    
                    # Calculate similarity: product of two Gaussian kernels
                    w_ij = np.exp(-color_dist_sq / (self.sigma_I ** 2)) * \
                           np.exp(-pos_dist_sq / (self.sigma_X ** 2))
                    
                    # Store non-zero values
                    if w_ij > 1e-6:
                        rows.append(i)
                        cols.append(j)
                        vals.append(w_ij)
        
        # Creating a sparse matrix
        W = sparse.csr_matrix((vals, (rows, cols)), shape=(n_pixels, n_pixels))
        
        print(f"Similarity matrix calculation completed, time: {time.time() - start_time:.2f}s")
        print(f"Matrix shape: {W.shape}, Non-zero elements: {W.nnz}, Density: {W.nnz/(n_pixels**2)*100:.6f}%")
        
        return W
    
    def _compute_normalized_laplacian(self, W):
        """
        Compute normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}.

        Parameters
        ----------
        W : scipy.sparse.csr_matrix
            Affinity matrix.

        Returns
        -------
        L : scipy.sparse.csr_matrix
            Normalized Laplacian.
        """

        print("Computing normalized Laplacian matrix...")
        start_time = time.time()
        
        # Calculate degree matrix
        d = np.array(W.sum(axis=1)).flatten()
        
        # Create a D^(-1/2) diagonal matrix
        d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d + 1e-10))
        
        # Compute the normalized Laplacian matrix
        L = sparse.identity(W.shape[0]) - d_inv_sqrt @ W @ d_inv_sqrt
        
        print(f"Laplacian matrix calculation completed, time: {time.time() - start_time:.2f}s")
        
        return L
    
    def _get_eigenvectors(self, L):
        """
        Extract the smallest non-zero eigenvectors of L.

        Parameters
        ----------
        L : scipy.sparse.csr_matrix
            Normalized Laplacian.

        Returns
        -------
        eigvecs : ndarray, shape (N, k)
            Eigenvectors corresponding to the 2nd through (k+1)-th smallest eigenvalues.
        """

        print(f"Computing {self.k} eigenvectors...")
        start_time = time.time()
        
        # Use the ARPACK solver to obtain the first k+1 eigenvalues and eigenvectors
        eigen_vals, eigen_vecs = linalg.eigsh(L, k=self.k+1, which='SM')
        
        # Sort the eigenvalues to ensure that the first k are the smallest non-zero eigenvalues
        idx = np.argsort(eigen_vals)
        eigen_vals = eigen_vals[idx]
        eigen_vecs = eigen_vecs[:, idx]
    
        # Use eigenvectors corresponding to the 2nd to (k+1)th smallest eigenvalues
        # (Skip the first one as it corresponds to constant vector with eigenvalue 0)
        eigen_vecs = eigen_vecs[:, 1:self.k+1]
        
        print(f"Eigenvector calculation completed, time: {time.time() - start_time:.2f}s")
        print(f"Eigenvalues: {eigen_vals[:self.k+1]}")
        
        return eigen_vecs
    
    def _kmeans_clustering(self, features, k):
        """
        Cluster rows of `features` into k groups via K-means.

        Parameters
        ----------
        features : ndarray, shape (N, k)
            Input feature vectors (e.g., eigenvectors).
        k : int
            Number of clusters.

        Returns
        -------
        labels : ndarray, shape (N,)
            Cluster labels [0..k-1].
        """

        try:
            # Try using the existing kmeans module
            from kmeans import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42)
        except ImportError:
            # If not found, use sklearn
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        
        print(f"Performing K-means clustering (k={k})...")
        start_time = time.time()
        
        # Normalize eigenvectors
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
        
        # K-means
        labels = kmeans.fit_predict(features_norm)
        
        print(f"K-means clustering completed, time: {time.time() - start_time:.2f}s")
        
        return labels
    
    def segment(self, image):
        """
        Perform full Normalized Cut segmentation on `image`.

        Parameters
        ----------
        image : ndarray, shape (H0, W0, C)
            Original input image.

        Returns
        -------
        segmented_resized : ndarray, shape (H0, W0)
            Integer segment labels for each pixel.
        """

        original_image_shape = image.shape[0:2]
        
        # Step 0: Preprocess the image
        processed_image = self._preprocess_image(image)
        
        # Step 1: Feature Extraction
        features, img_shape = self._extract_features(processed_image)
        
        # Step 2: Construct the similarity matrix W
        W = self._compute_similarity_matrix(features, img_shape)
        
        # Step 3: Calculate the normalized Laplacian matrix
        L = self._compute_normalized_laplacian(W)
        
        # Step 4: Solve for the first k eigenvectors
        eigen_vecs = self._get_eigenvectors(L)
        
        # Step 5: Perform K-means clustering on the eigenvectors
        labels = self._kmeans_clustering(eigen_vecs, self.k)
        
        # Reshape labels to image size
        h, w = img_shape
        segmented = labels.reshape(h, w)
        seg_resized = resize(
            segmented,
            original_image_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False
        )
        seg_resized = seg_resized.astype(segmented.dtype)
        return seg_resized
    
    def visualize_segmentation(self, image, segmented):
        """
        Display original & color-coded segmentation side by side.

        Parameters
        ----------
        image : ndarray, shape (H, W, C)
            Original image.
        segmented : ndarray, shape (H, W)
            Integer labels from `segment`.

        Returns
        -------
        colored : ndarray, shape (H, W, 3)
            RGB visualization of segments.
        """

        h, w = segmented.shape
        
        # Create a color map for the segmented regions
        colors = plt.cm.tab10(np.linspace(0, 1, self.k))
        segmented_color = np.zeros((h, w, 3))
        
        for i in range(self.k):
            segmented_color[segmented == i] = colors[i, :3]
        
        # Visualization
        plt.figure(figsize=(14, 6))
        
        plt.subplot(121)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(segmented_color)
        plt.title(f'Normalized Cut Segmentation (k={self.k})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return segmented_color
