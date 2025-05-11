import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.color import rgb2gray
from spectral_clustering import SpectralClustering

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

def calculate_iou(segmentation, ground_truth, class_idx, gt_class_idx):
    """
    Calculate the IOU between specific categories of two segmentation results
    """
    # Create binary masks
    mask_seg = (segmentation == class_idx)
    mask_gt = (ground_truth == gt_class_idx)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask_seg, mask_gt).sum()
    union = np.logical_or(mask_seg, mask_gt).sum()
    
    # Return IOU
    if union == 0:
        return 0.0
    return intersection / union

def find_best_matching_classes(segmentation, ground_truth, k=4):
    """
    Find the best matching category pairs between segmentation result and ground truth
    """
    matches = {}
    
    # Get all unique classes in ground truth
    gt_classes = np.unique(ground_truth)
    
    for i in range(k):
        best_iou = 0
        best_j = None
        
        for j in gt_classes:
            iou = calculate_iou(segmentation, ground_truth, i, j)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        
        matches[i] = (best_j, best_iou)
    
    return matches

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

def evaluate_with_ground_truth(image_path, gt_path, output_dir, params):
    """
    Evaluate Normalized Cut algorithm against ground truth
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
    
    # Read ground truth
    ground_truth = io.imread(gt_path, as_gray=True)
    ground_truth = resize(ground_truth, (64, 64), anti_aliasing=True, preserve_range=True)
    ground_truth = ground_truth.astype(np.int32)
    
    # Save original size
    original_size = f"{image.shape[0]}x{image.shape[1]}"
    
    # Preprocess image
    processed_image = preprocess_image(image, target_size=(64, 64))
    
    # Run Normalized Cut algorithm
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
    
    # Calculate IOU with ground truth
    print("Calculating IOU against ground truth...")
    
    # Find best matching classes for Normalized Cut compared to ground truth
    ncut_gt_matches = find_best_matching_classes(ncut_seg, ground_truth, params['k'])
    
    # Calculate average IOU
    ncut_avg_iou = sum(iou for _, iou in ncut_gt_matches.values()) / params['k']
    
    # Print results
    print(f"Normalized Cut average IOU: {ncut_avg_iou:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(image)
    plt.title(f'Original Image ({original_size})')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(ground_truth, cmap='tab10')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Create color maps
    ncut_color = plt.cm.tab10(ncut_seg % 10)[:,:,:3]
    
    plt.subplot(133)
    plt.imshow(ncut_color)
    plt.title(f'Normalized Cut\nTime: {ncut_time:.2f}s\nIOU: {ncut_avg_iou:.4f}')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save results
    image_name = os.path.basename(image_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{image_name}_evaluation.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Evaluation results have been saved to: {output_path}")
    
    return {
        'image_name': image_name,
        'ncut_time': ncut_time,
        'ncut_avg_iou': ncut_avg_iou,
        'ncut_gt_matches': ncut_gt_matches
    }

def generate_evaluation_report(results, output_dir):
    """
    Generate Normalized Cut evaluation report with ground truth comparison
    """
    image_names = [r['image_name'] for r in results]
    ncut_times = [r['ncut_time'] for r in results]
    ncut_ious = [r['ncut_avg_iou'] for r in results]
    
    # 1. Time performance chart
    plt.figure(figsize=(10, 6))
    plt.bar(image_names, ncut_times, color='royalblue')
    plt.xlabel('Image')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Normalized Cut: Execution Time')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save time chart
    time_plot_path = os.path.join(output_dir, "time_performance.png")
    plt.savefig(time_plot_path, dpi=300)
    plt.close()
    
    # 2. IOU evaluation chart
    plt.figure(figsize=(10, 6))
    plt.bar(image_names, ncut_ious, color='seagreen')
    plt.xlabel('Image')
    plt.ylabel('Average IOU (vs Ground Truth)')
    plt.title('Normalized Cut: IOU Comparison against Ground Truth')
    plt.ylim(0, 1)  # IOU range is 0-1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save IOU chart
    iou_plot_path = os.path.join(output_dir, "iou_evaluation.png")
    plt.savefig(iou_plot_path, dpi=300)
    plt.close()
    
    # 3. Generate summary text report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("===== Normalized Cut Algorithm Evaluation Report =====\n\n")
        f.write(f"Number of analyzed images: {len(results)}\n")
        f.write(f"Images: {', '.join(image_names)}\n\n")
        
        f.write("--- Performance Metrics ---\n")
        f.write(f"Average execution time: {np.mean(ncut_times):.2f} seconds\n")
        f.write(f"Average IOU against ground truth: {np.mean(ncut_ious):.4f}\n\n")
        
        f.write("--- Detailed Results for Each Image ---\n")
        for i, r in enumerate(results):
            f.write(f"Image {i+1}: {r['image_name']}\n")
            f.write(f"  Execution time: {r['ncut_time']:.2f} seconds\n")
            f.write(f"  Average IOU: {r['ncut_avg_iou']:.4f}\n")
            f.write("  Class matching with Ground Truth:\n")
            for ncut_idx, (gt_idx, iou) in r['ncut_gt_matches'].items():
                f.write(f"    Class {ncut_idx} → GT class {gt_idx}: IOU = {iou:.4f}\n")
            f.write("\n")
        
        f.write("--- Summary ---\n")
        f.write("The Normalized Cut algorithm considers the global structure of the image, which can lead to more meaningful segmentation results compared to methods that only consider local features.\n")
        f.write(f"Based on the evaluation, the algorithm achieves an average IOU of {np.mean(ncut_ious):.4f} against ground truth segmentation.\n")
        f.write(f"The average execution time is {np.mean(ncut_times):.2f} seconds per image, which indicates the computational complexity of the algorithm.\n")
    
    print(f"Evaluation report has been saved to: {report_path}")
    print(f"Time performance chart has been saved to: {time_plot_path}")
    print(f"IOU evaluation chart has been saved to: {iou_plot_path}")

def main():
    """
    Main function: Run segmentation and evaluation
    """
    # Set the data directory and output directory
    data_dir = "./data"
    gt_dir = "./ground_truth"  # Ground truth directory
    output_dir = "./output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read test images (from CIFAR-10)
    image_files = ["horse.png", "deer.png", "airplane.png"]
    image_paths = [os.path.join(data_dir, img) for img in image_files]
    gt_paths = [os.path.join(gt_dir, img.replace('.png', '_gt.png')) for img in image_files]
    
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
    print("2. Algorithm evaluation with ground truth")
    print("3. Run both functions")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == '1' or choice == '3':
        print("\n=== Running Basic Image Segmentation ===")
        for image_path in image_paths:
            if os.path.exists(image_path):
                process_single_image(image_path, output_dir, params, preprocess_params)
            else:
                print(f"Warning: Image {image_path} does not exist, skipping")
    
    if choice == '2' or choice == '3':
        print("\n=== Running Algorithm Evaluation with Ground Truth ===")
        # Check if files exist
        valid_pairs = []
        for img_path, gt_path in zip(image_paths, gt_paths):
            if os.path.exists(img_path) and os.path.exists(gt_path):
                valid_pairs.append((img_path, gt_path))
            else:
                if not os.path.exists(img_path):
                    print(f"Warning: Image {img_path} does not exist, skipping")
                elif not os.path.exists(gt_path):
                    print(f"Warning: Ground truth {gt_path} does not exist, skipping")
        
        if not valid_pairs:
            print("No valid image-ground truth pairs found! Please prepare the data first.")
            return
        
        # Evaluate each image with ground truth
        results = []
        for img_path, gt_path in valid_pairs:
            try:
                result = evaluate_with_ground_truth(img_path, gt_path, output_dir, params)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating image {img_path}: {e}")
        
        # Generate evaluation report
        if results:
            generate_evaluation_report(results, output_dir)
        else:
            print("No successful evaluations, cannot generate report.")

if __name__ == "__main__":
    main()
