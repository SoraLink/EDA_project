import abc

import numpy as np


class ImageSegmentAlgorithm(abc.ABC):
    
    @abc.abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment the input image using a specific segmentation algorithm.

        This method should be implemented by subclasses to perform image segmentation.
        Supported approaches may include K-means clustering, spectral clustering,
        or graph-based superpixel clustering.

        Parameters
        ----------
        image : np.ndarray
            The image file to be segmented.

        Returns
        -------
        np.ndarray
            A 2D array of shape (H, W), where H and W are the height and width of the image.
            Each element represents the segment label of the corresponding pixel.
        """
        raise NotImplementedError
    
    