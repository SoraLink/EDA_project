import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.color import rgb2gray
from spectral_clustering import SpectralClustering
import time

def preprocess_image(image, target_size=(64, 64), use_grayscale=False, apply_smoothing=True):
    """
    Preprocess images: resize, optionally convert to grayscale and apply smoothing filters
    """
    # Scaling an image
    if target_size is not None:
        image = resize(image, target_size, anti_aliasing=True)
    
    # Convert to Grayscale
    if use_grayscale and len(image.shape) > 2:
        image = rgb2gray(image)
        # Convert back to 3 channels to be compatible with the algorithm
        image = np.stack([image] * 3, axis=2)
    
    # Apply Gaussian smoothing
    if apply_smoothing:
        # Apply smoothing to each channel separately
        for i in range(image.shape[2]):
            image[:,:,i] = gaussian(image[:,:,i], sigma=1)
    
    return image

def calculate_iou(segmentation1, segmentation2, class_idx1, class_idx2):
    """
    Calculate the IOU between specific categories of two segmentation results
    """
    # Create binary masks
    mask1 = (segmentation1 == class_idx1)
    mask2 = (segmentation2 == class_idx2)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Return IOU
    if union == 0:
        return 0.0
    return intersection / union

def find_best_matching_classes(ncut_seg, kmeans_seg, k=4):
    """
    Find the best matching category pairs between two segmentation results
    """
    matches = {}
    
    for i in range(k):
        best_iou = 0
        best_j = None
        
        for j in range(k):
            iou = calculate_iou(ncut_seg, kmeans_seg, i, j)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        
        matches[i] = (best_j, best_iou)
    
    return matches

def kmeans_segmentation(image, k=4):
    """
    Use K-means for basic image segmentation
    """
    try:
        # Try to use custom KMeans
        from kmeans import KMeans
        kmeans = KMeans(n_clusters=k)
    except:
        # Otherwise use sklearn
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    h, w, c = image.shape
    flat_image = image.reshape(-1, c)
    
    # Apply K-means
    labels = kmeans.fit_predict(flat_image)
    
    # Reshape to image size
    return labels.reshape(h, w)

def process_single_image(image_path, output_dir, params, preprocess_params):
    """
    Process a single image with Normalized Cut segmentation
    """
    if not os.path.exists(image_path):
        print(f"The image file does not exist: {image_path}")
        return None
        
    print(f"Processing image: {image_path}")
    
    # Reading an Image
    image = io.imread(image_path)
    if len(image.shape) == 2:  # Grayscale to RGB
        image = np.stack([image] * 3, axis=2)
    elif image.shape[2] > 3:  # Remove Alpha Channel
        image = image[:, :, :3]
    
    # Save original image size
    original_size = f"{image.shape[0]}x{image.shape[1]}"
    
    # Apply image preprocessing
    processed_image = preprocess_image(
        image,
        target_size=preprocess_params['target_size'],
        use_grayscale=preprocess_params['use_grayscale'],
        apply_smoothing=preprocess_params['apply_smoothing']
    )
    
    # Creating an Algorithm Instance
    spectral = SpectralClustering(
        sigma_I=params['sigma_I'],
        sigma_X=params['sigma_X'],
        r=params['r'],
        k=params['k']
    )
    
    print(f"Start image segmentation...")
    
    # segmentation
    segmented = spectral.segment(processed_image)
    
    # Creating visualization results
    h, w = segmented.shape
    colors = plt.cm.tab10(np.linspace(0, 1, params['k']))
    segmented_color = np.zeros((h, w, 3))
    
    for i in range(params['k']):
        segmented_color[segmented == i] = colors[i, :3]
    
    # Save segmentation results
    image_name = os.path.basename(image_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{image_name}_segmented.png")
    
    plt.figure(figsize=(16, 8))
    
    plt.subplot(131)
    plt.imshow(image)
    plt.title(f'Original Image ({original_size})')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(processed_image)
    plt.title(f'Preprocessing images ({preprocess_params["target_size"][0]}x{preprocess_params["target_size"][1]})')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(segmented_color)
    plt.title(f'Normalized Cut (k={params["k"]})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"The segmentation results have been saved to: {output_path}")
    
    return segmented, processed_image

def evaluate_algorithm_comparison(image_path, output_dir, params):
    """
    Evaluate and compare Normalized Cut and K-means algorithms on a given image
    """
    print(f"\nEvaluating image: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read image
    image = io.imread(image_path)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=2)
    elif image.shape[2] > 3:
        image = image[:, :, :3]
    
    # Save original size
    original_size = f"{image.shape[0]}x{image.shape[1]}"
    
    # Preprocess image
    processed_image = preprocess_image(image, target_size=(64, 64))
    
    # 1. Run Normalized Cut algorithm
    print("Running Normalized Cut algorithm...")
    start_time = time.time()
    spectral = SpectralClustering(
        sigma_I=params['sigma_I'],
        sigma_X=params['sigma_X'],
        r=params['r'],
        k=params['k']
    )
    ncut_seg = spectral.segment(processed_image)
    ncut_time = time.time() - start_time
    print(f"Normalized Cut time: {ncut_time:.2f} seconds")
    
    # 2. Run K-means algorithm
    print("Running K-means algorithm...")
    start_time = time.time()
    kmeans_seg = kmeans_segmentation(processed_image, params['k'])
    kmeans_time = time.time() - start_time
    print(f"K-means time: {kmeans_time:.2f} seconds")
    
    # 3. Calculate IOU
    print("Calculating IOU...")
    matches = find_best_matching_classes(ncut_seg, kmeans_seg, params['k'])
    
    # Calculate average IOU
    avg_iou = sum(iou for _, iou in matches.values()) / params['k']
    
    # Print results
    print(f"Average IOU: {avg_iou:.4f}")
    print("Category matching and IOU:")
    for ncut_idx, (kmeans_idx, iou) in matches.items():
        print(f"  Normalized Cut category {ncut_idx} corresponds to K-means category {kmeans_idx}, IOU: {iou:.4f}")
    
    # Visualize results
    plt.figure(figsize=(16, 10))
    
    plt.subplot(131)
    plt.imshow(image)
    plt.title(f'Original Image ({original_size})')
    plt.axis('off')
    
    # Create color maps
    ncut_color = plt.cm.tab10(ncut_seg % 10)[:,:,:3]
    kmeans_color = plt.cm.tab10(kmeans_seg % 10)[:,:,:3]
    
    plt.subplot(132)
    plt.imshow(ncut_color)
    plt.title(f'Normalized Cut\nTime: {ncut_time:.2f}s')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(kmeans_color)
    plt.title(f'K-means\nTime: {kmeans_time:.2f}s')
    plt.axis('off')
    
    plt.suptitle(f'Segmentation Comparison (Average IOU: {avg_iou:.4f})', fontsize=16)
    plt.tight_layout()
    
    # Save results
    image_name = os.path.basename(image_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{image_name}_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Comparison results have been saved to: {output_path}")
    
    return {
        'image_name': image_name,
        'ncut_time': ncut_time,
        'kmeans_time': kmeans_time,
        'avg_iou': avg_iou,
        'matches': matches
    }

def generate_comparison_report(results, output_dir):
    """
    Generate algorithm comparison report
    """
    image_names = [r['image_name'] for r in results]
    ncut_times = [r['ncut_time'] for r in results]
    kmeans_times = [r['kmeans_time'] for r in results]
    ious = [r['avg_iou'] for r in results]
    
    # 1. Time comparison chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(image_names))
    width = 0.35
    
    plt.bar(x - width/2, ncut_times, width, label='Normalized Cut')
    plt.bar(x + width/2, kmeans_times, width, label='K-means')
    
    plt.xlabel('Image')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Normalized Cut vs K-means: Execution Time Comparison')
    plt.xticks(x, image_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save time comparison chart
    time_plot_path = os.path.join(output_dir, "time_comparison.png")
    plt.savefig(time_plot_path, dpi=300)
    plt.close()
    
    # 2. IOU comparison chart
    plt.figure(figsize=(10, 6))
    plt.bar(image_names, ious, color='skyblue')
    plt.xlabel('Image')
    plt.ylabel('Average IOU')
    plt.title('Average IOU between Normalized Cut and K-means')
    plt.ylim(0, 1)  # IOU range is 0-1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save IOU comparison chart
    iou_plot_path = os.path.join(output_dir, "iou_comparison.png")
    plt.savefig(iou_plot_path, dpi=300)
    plt.close()
    
    # 3. Generate summary text report
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("===== Image Segmentation Algorithm Comparison Report =====\n\n")
        f.write(f"Number of analyzed images: {len(results)}\n")
        f.write(f"Images: {', '.join(image_names)}\n\n")
        
        f.write("--- Execution Time Comparison ---\n")
        f.write(f"Normalized Cut average time: {np.mean(ncut_times):.2f} seconds\n")
        f.write(f"K-means average time: {np.mean(kmeans_times):.2f} seconds\n")
        f.write(f"Time ratio (Normalized Cut / K-means): {np.mean(ncut_times)/np.mean(kmeans_times):.2f}x\n\n")
        
        f.write("--- IOU Comparison ---\n")
        f.write(f"Average IOU: {np.mean(ious):.4f}\n\n")
        
        f.write("--- Detailed Results for Each Image ---\n")
        for i, r in enumerate(results):
            f.write(f"Image {i+1}: {r['image_name']}\n")
            f.write(f"  Normalized Cut time: {r['ncut_time']:.2f} seconds\n")
            f.write(f"  K-means time: {r['kmeans_time']:.2f} seconds\n")
            f.write(f"  Average IOU: {r['avg_iou']:.4f}\n")
            f.write("  Category matching:\n")
            for ncut_idx, (kmeans_idx, iou) in r['matches'].items():
                f.write(f"    Normalized Cut category {ncut_idx} → K-means category {kmeans_idx}: IOU = {iou:.4f}\n")
            f.write("\n")
        
        f.write("--- Summary ---\n")
        if np.mean(ncut_times) > np.mean(kmeans_times):
            f.write(f"K-means algorithm is {np.mean(ncut_times)/np.mean(kmeans_times):.2f} times faster than Normalized Cut in execution speed.\n")
        else:
            f.write(f"Normalized Cut algorithm is {np.mean(kmeans_times)/np.mean(ncut_times):.2f} times faster than K-means in execution speed.\n")
        
        f.write("The advantage of the Normalized Cut algorithm is that it considers the global structure of the image, while K-means only considers color similarity.\n")
        f.write("The average IOU value indicates that there is some similarity between the segmentations produced by the two algorithms, but also significant differences.\n")
    
    print(f"Comparison report has been saved to: {report_path}")
    print(f"Time comparison chart has been saved to: {time_plot_path}")
    print(f"IOU comparison chart has been saved to: {iou_plot_path}")

def main():
    """
    Main function: Run segmentation and comparison
    """
    # Set the data directory and output directory
    data_dir = "./data"
    output_dir = "./output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read test images (from CIFAR-10)
    image_paths = [
        os.path.join(data_dir, "horse.png"),
        os.path.join(data_dir, "deer.png"),
        os.path.join(data_dir, "airplane.png")
    ]
    
    # parameter
    params = {
        'sigma_I': 0.1,  # Color similarity parameter
        'sigma_X': 4.0,  # Spatial distance parameter
        'r': 5,          # Neighborhood radius
        'k': 4           # Number of segmentation categories
    }
    
    # Image preprocessing parameters
    preprocess_params = {
        'target_size': (64, 64),  
        'use_grayscale': False,   # Whether to convert to grayscale
        'apply_smoothing': True   # Whether to apply smoothing
    }
    
    print("Please select the function to run:")
    print("1. Basic image segmentation")
    print("2. Algorithm evaluation and comparison")
    print("3. Run both functions")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == '1' or choice == '3':
        print("\n=== Running Basic Image Segmentation ===")
        for image_path in image_paths:
            process_single_image(image_path, output_dir, params, preprocess_params)
    
    if choice == '2' or choice == '3':
        print("\n=== Running Algorithm Evaluation and Comparison ===")
        # Check if files exist
        existing_paths = [p for p in image_paths if os.path.exists(p)]
        if not existing_paths:
            print("No image files found! Please run prepare_data.py to prepare the data first.")
            return
        
        # Evaluate each image
        results = []
        for image_path in existing_paths:
            result = evaluate_algorithm_comparison(image_path, output_dir, params)
            results.append(result)
        
        # Generate comparison report
        generate_comparison_report(results, output_dir)

if __name__ == "__main__":
    main()