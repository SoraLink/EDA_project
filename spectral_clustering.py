import abc
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
import time
from image_segement_algorithm import ImageSegmentAlgorithm

class SpectralClustering(ImageSegmentAlgorithm):
    """
    Implementation of image segmentation using Normalized Cut algorithm
    Reference: Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation.
    """
    
    def __init__(self, sigma_I=0.05, sigma_X=4.0, r=5, k=6):

        self.sigma_I = sigma_I
        self.sigma_X = sigma_X
        self.r = r
        self.k = k
    
    def _extract_features(self, image):
        h, w, c = image.shape
        # Normalize the image to [0,1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        # Creating a Coordinate Grid
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Flatten the features into an NxC matrix, where N is the number of pixels and C is the feature dimension
        features = np.zeros((h*w, c+2), dtype=np.float32)
        features[:, :c] = image.reshape(-1, c)        # Color Features
        features[:, c] = x_coords.flatten() / w       # x coordinate (normalized)
        features[:, c+1] = y_coords.flatten() / h     # y coordinate (normalized)
        
        return features, (h, w)
    
    def _compute_similarity_matrix(self, features, img_shape):
        print("Calculate the similarity matrix...")
        start_time = time.time()
        
        h, w = img_shape
        n_pixels = h * w
        
        # Create a list of rows, columns, and values ​​for a sparse matrix
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
        
        print(f"Similarity matrix calculation completed, time consuming: {time.time() - start_time:.2f}s")
        print(f"Matrix shape: {W.shape}, Number of non-zero elements: {W.nnz}, Density: {W.nnz/(n_pixels**2)*100:.6f}%")
        
        return W
    
    def _compute_normalized_laplacian(self, W):
        """Normalized Laplacian matrix: L = D^(-1/2) * (D - W) * D^(-1/2)"""
        print("Normalized Laplacian matrix...")
        start_time = time.time()
        
        # Calculate degree matrix
        d = np.array(W.sum(axis=1)).flatten()
        
        # Create a D^(-1/2) diagonal matrix
        d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d + 1e-10))
        
        # Compute the normalized Laplacian matrix
        L = sparse.identity(W.shape[0]) - d_inv_sqrt @ W @ d_inv_sqrt
        
        print(f"Laplace matrix calculation completed, time consuming: {time.time() - start_time:.2f}s")
        
        return L
    
    def _get_eigenvectors(self, L):
        print(f"Calculate the number of eigenvectors before{self.k}...")
        start_time = time.time()
        
        # Use the ARPACK solver to obtain the first k+1 eigenvalues ​​and eigenvectors
        eigen_vals, eigen_vecs = linalg.eigsh(L, k=self.k+1, which='SM')
        
        # Sort the eigenvalues ​​to ensure that the first k are the smallest non-zero eigenvalues
        idx = np.argsort(eigen_vals)
        eigen_vals = eigen_vals[idx]
        eigen_vecs = eigen_vecs[:, idx]
    
        eigen_vecs = eigen_vecs[:, 1:self.k+1]
        
        print(f"Eigenvector calculation completed, time consuming: {time.time() - start_time:.2f}s")
        print(f"Minimum eigenvalue: {eigen_vals[:self.k+1]}")
        
        return eigen_vecs
    
    def _kmeans_clustering(self, features, k):
        """K-means clustering of feature vectors"""
        try:
            # Try using the existing kmeans module
            from kmeans import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42)
        except:
            # If not found, sklearn is used
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        
        print(f"Perform K-means clustering(k={k})...")
        start_time = time.time()
        
        # Normalized eigenvector
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
        
        # K-means
        labels = kmeans.fit_predict(features_norm)
        
        print(f"K-means clustering completed, time consuming: {time.time() - start_time:.2f}s")
        
        return labels
    
    def segment(self, image):
        """
        Implementing abstract methods: performing image segmentation
        """
        overall_start = time.time()
        
        # Step 1: Feature Extraction
        features, img_shape = self._extract_features(image)
        
        # Step 2: Construct the similarity matrix W
        W = self._compute_similarity_matrix(features, img_shape)
        
        # Step 3: Calculate the normalized Laplacian matrix
        L = self._compute_normalized_laplacian(W)
        
        # Step 4: Solve for the first k eigenvectors
        eigen_vecs = self._get_eigenvectors(L)
        
        # Step 5: Perform K-means clustering on the feature vector
        labels = self._kmeans_clustering(eigen_vecs, self.k)
        
        # Reshape labels to image size
        h, w = img_shape
        segmented = labels.reshape(h, w)
        
        print(f"Total time spent on segmentation: {time.time() - overall_start:.2f}s")
        
        return segmented
    
    def visualize_results(self, image, segmented):
        """Visualizing segmentation results"""
        h, w = segmented.shape
        
        # Create a new color map to the segmented regions
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
        plt.title(f'Normalized Cut segmentation results (k={self.k})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return segmented_color