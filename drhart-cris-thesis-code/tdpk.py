import sys
import os
import torch
import torchvision.transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.segmentation import deeplabv3_resnet101
import matplotlib.pyplot as plt
import numpy as np
import logging
from PIL import Image
import cv2
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import map_coordinates
import warnings

# Suppress specific warnings if necessary
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# iPad 12.9-inch 6th generation camera parameters (estimated)
IPAD_FOCAL_LENGTH = 5.9  # mm
IPAD_SENSOR_WIDTH = 5.6  # mm
IPAD_SENSOR_HEIGHT = 4.2  # mm

# Conversion factor: 1 meter = 39.3701 inches
METER_TO_INCHES = 39.3701

# Keypoint indices for sideseam and waistline
SIDESEAM_INDICES = [11, 12, 13, 14]
# RIGHT_SIDESEAM_INDICES = [11, 12, 13, 14]  # Right sideseam keypoints
# LEFT_SIDESEAM_INDICES = [16, 17, 18, 19]  # Left sideseam keypoints
WAISTLINE_INDICES = [14, 15, 16]

# DrHartCrisDataset class for loading image and depth files
class DrHartCrisDataset(Dataset):
    def __init__(self, image_dirs, depth_dirs):
        self.data = []
        for image_dir, depth_dir in zip(image_dirs, depth_dirs):
            image_path = Path(image_dir)
            if not image_path.exists():
                logger.warning(f"Image directory not found: {image_path}")
                continue

            for img_file in image_path.glob('*.png'):
                # Assuming depth files are named as 'depth_{last_two_chars_of_image_stem}.npy'
                depth_file = os.path.join(depth_dir, f'depth_{img_file.stem[-2:]}.npy')
                if os.path.exists(img_file) and os.path.exists(depth_file):
                    self.data.append((img_file, depth_file))
                else:
                    if not os.path.exists(img_file):
                        logger.warning(f"Image file not found: {img_file}")
                    if not os.path.exists(depth_file):
                        logger.warning(f"Depth file not found: {depth_file}")
        if not self.data:
            logger.warning("No data found in the provided directories.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, depth_file = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        depth_map = np.load(depth_file)

        return {
            'image': image,
            'depth_map': depth_map
        }

# Load KeypointRCNN model
def load_keypoint_model(checkpoint_path):
    model = keypointrcnn_resnet50_fpn(pretrained=False, num_classes=14, num_keypoints=39)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state_dict = torch.load(checkpoint_path, map_location=device)

    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    num_keypoints = 39
    model.roi_heads.keypoint_predictor.kps_score_lowres = torch.nn.Conv2d(in_features, num_keypoints, kernel_size=1, stride=1)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    logger.info("Keypoint model loaded successfully.")
    return model

# Load Segmentation model
def load_segmentation_model():
    segmentation_model = deeplabv3_resnet101(weights='DEFAULT')
    segmentation_model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    segmentation_model.to(device)
    logger.info("Segmentation model loaded successfully.")
    return segmentation_model, device

# Calculate the Euclidean distance between keypoints
def calculate_distance(keypoints):
    if keypoints.shape[0] < 2:
        return 0.0
    return np.sum(np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=1))

def predict_keypoints(model, image):
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract the predicted keypoints and labels
    keypoints = predictions[0]['keypoints']
    labels = predictions[0]['labels']

    # Filter keypoints specific to the "short sleeve top" (assuming category ID 1)
    short_sleeve_top_indices = (labels == 1).nonzero(as_tuple=True)[0]

    if len(short_sleeve_top_indices) > 0:
        # Get the first detected short sleeve top's keypoints
        short_sleeve_top_keypoints = keypoints[short_sleeve_top_indices[0]].cpu().numpy()[:25]  # Limit to first 25 keypoints
        logger.debug("Keypoints predicted successfully.")
        return short_sleeve_top_keypoints
    else:
        logger.warning("No short sleeve top detected in the image.")
        return np.array([])

def get_segmentation_mask(model, device, image):
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = torch.argmax(output, dim=0).cpu().numpy()

    person_class = 15  # COCO dataset class index for 'person' in DeepLabV3
    binary_mask = (mask == person_class).astype(np.uint8)
    logger.debug("Segmentation mask generated.")
    return binary_mask

# Get depth value with subpixel precision
def get_subpixel_depth(x, y, depth_map):
    return map_coordinates(depth_map, [[y], [x]], order=1, mode='nearest')[0]

# Convert pixel-space keypoints to real-space coordinates
def pixel_to_real_space(keypoints, depth_map, image_width, image_height):
    fx = (IPAD_FOCAL_LENGTH / IPAD_SENSOR_WIDTH) * image_width
    fy = (IPAD_FOCAL_LENGTH / IPAD_SENSOR_HEIGHT) * image_height
    cx, cy = image_width / 2, image_height / 2

    depth_height, depth_width = depth_map.shape
    scale_x = depth_width / image_width
    scale_y = depth_height / image_height

    real_space_keypoints = []
    for kp in keypoints:
        x, y, z = kp
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        z = get_subpixel_depth(scaled_x, scaled_y, depth_map)
        real_x = (x - cx) * z / fx
        real_y = (y - cy) * z / fy
        real_space_keypoints.append([real_x, real_y, z])

    logger.debug("Converted keypoints to real-world coordinates using absolute depth.")
    return np.array(real_space_keypoints)

# Adjust keypoints to lie within the segmentation mask and apply depth filtering
def adjust_keypoints_with_segmentation_and_depth(keypoints, segmentation_mask, depth_map, depth_threshold):
    refined_keypoints = []

    assert segmentation_mask.shape == depth_map.shape, "Segmentation mask and depth map must have the same dimensions"

    for kp in keypoints:
        x, y, z = kp
        x_clipped = np.clip(x, 0, segmentation_mask.shape[1] - 1)
        y_clipped = np.clip(y, 0, segmentation_mask.shape[0] - 1)

        z = get_subpixel_depth(x_clipped, y_clipped, depth_map)

        if z > depth_threshold or segmentation_mask[int(y_clipped), int(x_clipped)] == 0:
            x_new, y_new = adjust_keypoint_closer(x_clipped, y_clipped, segmentation_mask)
            z_new = get_subpixel_depth(x_new, y_new, depth_map)
            refined_keypoints.append([x_new, y_new, z_new])
            logger.debug(f"Keypoint adjusted from ({x}, {y}, {z:.4f}m) to ({x_new}, {y_new}, {z_new:.4f}m)")
        else:
            refined_keypoints.append([x_clipped, y_clipped, z])

    return np.array(refined_keypoints)

# Find the nearest valid point within the segmentation mask
def adjust_keypoint_closer(x, y, segmentation_mask):
    mask_points = np.column_stack(np.where(segmentation_mask > 0))
    if len(mask_points) == 0:
        logger.warning("Segmentation mask is empty. Cannot adjust keypoints.")
        return x, y

    distances = np.linalg.norm(mask_points - np.array([y, x]), axis=1)
    nearest_idx = np.argmin(distances)
    nearest_point = mask_points[nearest_idx]
    return nearest_point[1], nearest_point[0]  # (x, y)

# Enforce depth consistency across keypoints
def enforce_depth_consistency(keypoints, groups, max_depth_diff=0.1):
    adjusted_keypoints = keypoints.copy()
    for group in groups:
        group_depths = adjusted_keypoints[group, 2]
        median_depth = np.median(group_depths)
        for idx in group:
            if abs(adjusted_keypoints[idx, 2] - median_depth) > max_depth_diff:
                old_depth = adjusted_keypoints[idx, 2]
                adjusted_keypoints[idx, 2] = median_depth
                logger.debug(f"Keypoint {idx} depth adjusted from {old_depth:.4f}m to {median_depth:.4f}m for consistency.")
    return adjusted_keypoints

def refine_keypoints_local_depth(keypoints, depth_map, window_size=5):
    refined_keypoints = []
    h, w = depth_map.shape
    for kp in keypoints:
        x, y, _ = kp
        x, y = int(x), int(y)
        x1, x2 = max(0, x - window_size), min(w, x + window_size + 1)
        y1, y2 = max(0, y - window_size), min(h, y + window_size + 1)
        local_depth = depth_map[y1:y2, x1:x2]
        local_median = np.median(local_depth)
        best_x, best_y = x, y
        min_diff = float('inf')
        for dy in range(y1, y2):
            for dx in range(x1, x2):
                if abs(depth_map[dy, dx] - local_median) < min_diff:
                    min_diff = abs(depth_map[dy, dx] - local_median)
                    best_x, best_y = dx, dy
        refined_keypoints.append([best_x, best_y, depth_map[best_y, best_x]])
    return np.array(refined_keypoints)

# Calculate the Euclidean distance between keypoints
def calculate_distance(keypoints):
    if keypoints.shape[0] < 2:
        return 0.0
    return np.sum(np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=1))

# Process sample, predict keypoints, and calculate distances
def process_sample(sample, keypoint_model, segmentation_model, seg_device):
    image = sample['image']
    depth_map = sample['depth_map']

    predicted_keypoints = predict_keypoints(keypoint_model, image)

    # If no keypoints for short sleeve top, return empty result
    if len(predicted_keypoints) == 0:
        return None

    image_width, image_height = image.size
    depth_map_resized = cv2.resize(depth_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    segmentation_mask = get_segmentation_mask(segmentation_model, seg_device, image)
    segmentation_mask_resized = cv2.resize(segmentation_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

    refined_keypoints = refine_keypoints_local_depth(predicted_keypoints, depth_map_resized)
    adjusted_keypoints = adjust_keypoints_with_segmentation_and_depth(refined_keypoints, segmentation_mask_resized, depth_map_resized, depth_threshold=1.0)
    real_space_keypoints = pixel_to_real_space(adjusted_keypoints, depth_map_resized, image_width, image_height)

    # Apply depth consistency to grouped keypoints
    keypoint_groups = [SIDESEAM_INDICES, WAISTLINE_INDICES]
    final_keypoints = enforce_depth_consistency(real_space_keypoints, keypoint_groups, max_depth_diff=0.1)

    sideseam_distance_meters = calculate_distance(final_keypoints[SIDESEAM_INDICES])
    waistline_distance_meters = calculate_distance(final_keypoints[WAISTLINE_INDICES])

    sideseam_distance_inches = sideseam_distance_meters * METER_TO_INCHES
    waistline_distance_inches = waistline_distance_meters * METER_TO_INCHES

    logger.info("Sample processed successfully.")
    return {
        'image': image,
        'predicted_keypoints': adjusted_keypoints,
        'real_space_keypoints': final_keypoints,
        'depth_map': depth_map_resized,
        'segmentation_mask': segmentation_mask_resized,
        'sideseam_distance_meters': sideseam_distance_meters,
        'waistline_distance_meters': waistline_distance_meters,
        'sideseam_distance_inches': sideseam_distance_inches,
        'waistline_distance_inches': waistline_distance_inches
    }

# Visualize results: image with keypoints, depth map, segmentation mask, and 3D keypoints
def visualize_results(image, keypoints, results):
    fig = plt.figure(figsize=(20, 15))

    # Plot the original image with predicted keypoints
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(np.array(image))
    if keypoints is not None and len(keypoints) > 0:
        # Ensure keypoints are in pixel coordinates
        pixel_keypoints = keypoints[:, :2]  # Only use x and y coordinates
        ax1.scatter(pixel_keypoints[:, 0], pixel_keypoints[:, 1], c='r', s=20)  # Plot keypoints

        # Add labels for each keypoint
        for i, (x, y) in enumerate(pixel_keypoints):
            ax1.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                         fontsize=8, color='white', backgroundcolor='black')

    ax1.set_title('Image with Predicted Keypoints')

    # Plot lines for Sideseam and Waistline keypoints
    if len(keypoints) > max(SIDESEAM_INDICES + WAISTLINE_INDICES):
        sideseam = pixel_keypoints[SIDESEAM_INDICES]
        waistline = pixel_keypoints[WAISTLINE_INDICES]

        ax1.plot(sideseam[:, 0], sideseam[:, 1], 'b-', lw=2, label='Sideseam')
        ax1.plot(waistline[:, 0], waistline[:, 1], 'g-', lw=2, label='Waistline')

        ax1.legend()

    # Depth map visualization
    ax2 = fig.add_subplot(2, 3, 2)
    depth_map = results['depth_map']
    depth_map_display = depth_map.copy()
    depth_map_display[depth_map_display == 0] = np.nan  # Optional: Handle zero depth values
    im = ax2.imshow(depth_map_display, cmap='plasma')
    ax2.set_title('Depth Map')
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='Depth (meters)')

    # Segmentation mask visualization
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(results['segmentation_mask'], cmap='gray')
    ax3.set_title('Segmentation Mask')

    # 3D plot for keypoints in real-world space
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    real_space_keypoints = results['real_space_keypoints']
    ax4.scatter(real_space_keypoints[:, 0], real_space_keypoints[:, 1], real_space_keypoints[:, 2], c='r', s=20)

    # Add labels for each keypoint in 3D space
    for i, (x, y, z) in enumerate(real_space_keypoints):
        ax4.text(x, y, z, str(i), fontsize=8)

    # Plot lines for sideseams and waistline in 3D
    sideseam_3d = real_space_keypoints[SIDESEAM_INDICES]
    waistline_3d = real_space_keypoints[WAISTLINE_INDICES]

    ax4.plot(sideseam_3d[:, 0], sideseam_3d[:, 1], sideseam_3d[:, 2], 'b-', linewidth=2, label='Sideseam')
    ax4.plot(waistline_3d[:, 0], waistline_3d[:, 1], waistline_3d[:, 2], 'g-', linewidth=2, label='Waistline')

    ax4.set_xlabel('X (meters)')
    ax4.set_ylabel('Y (meters)')
    ax4.set_zlabel('Z (meters)')
    ax4.set_title('3D Keypoints')
    ax4.legend()

    # Overlay segmentation mask on the depth map
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(depth_map, cmap='plasma')
    ax5.imshow(results['segmentation_mask'], cmap='jet', alpha=0.3)
    ax5.set_title('Depth Map with Segmentation Mask')

    plt.tight_layout()
    plt.show()

    # Print calculated distances
    print(f"Sideseam Distance: {results['sideseam_distance_meters']:.4f} meters ({results['sideseam_distance_inches']:.2f} inches)")
    print(f"Waistline Distance: {results['waistline_distance_meters']:.4f} meters ({results['waistline_distance_inches']:.2f} inches)")

    # Print depths for each keypoint
    print("\nKeypoint Depths:")
    for i, kp in enumerate(results['real_space_keypoints']):
        print(f"Keypoint {i}: {kp[2]:.4f} meters")

# Main function to orchestrate the processing
def main():
    # Load the custom-trained KeypointRCNN model
    keypoint_model = load_keypoint_model(r"C:\Users\crisz\Desktop\Prod-Thesis\checkpoints\deepfashion2_model.pth")

    # Load the Segmentation model
    segmentation_model, seg_device = load_segmentation_model()

    # Directories for Cris-run
    image_dirs_cris = [r"C:\Users\crisz\Desktop\Prod-Thesis\DrHart&CrisData\cris-run\lidar\run1\image"]
    depth_dirs_cris = [r"C:\Users\crisz\Desktop\Prod-Thesis\DrHart&CrisData\cris-run\lidar\run1\depth"]

    # Directories for David-run
    image_dirs_david = [r"C:\Users\crisz\Desktop\Prod-Thesis\DrHart&CrisData\david-run\lidar\run1\image"]
    depth_dirs_david = [r"C:\Users\crisz\Desktop\Prod-Thesis\DrHart&CrisData\david-run\lidar\run1\depth"]

    # Initialize datasets for Cris-run and David-run
    dataset_cris = DrHartCrisDataset(image_dirs_cris, depth_dirs_cris)
    dataset_david = DrHartCrisDataset(image_dirs_david, depth_dirs_david)

    if len(dataset_cris) == 0 and len(dataset_david) == 0:
        logger.error("No data found in both Cris-run and David-run directories. Exiting.")
        return

    # Process Cris-run samples
    if len(dataset_cris) > 0:
        logger.info("Processing Cris-run samples...")
        for i in range(len(dataset_cris)):
            sample_cris = dataset_cris[i]
            results_cris = process_sample(sample_cris, keypoint_model, segmentation_model, seg_device)
            if results_cris:
                visualize_results(sample_cris['image'], results_cris['predicted_keypoints'], results_cris)

            user_input = input("Press Enter to continue to the next Cris-run sample, or 'q' to quit: ")
            if user_input.lower() == 'q':
                logger.info("Process terminated by user during Cris-run processing.")
                break

    # Process David-run samples
    if len(dataset_david) > 0:
        logger.info("Processing David-run samples...")
        for i in range(len(dataset_david)):
            sample_david = dataset_david[i]
            results_david = process_sample(sample_david, keypoint_model, segmentation_model, seg_device)
            if results_david:
                visualize_results(sample_david['image'], results_david['predicted_keypoints'], results_david)

            user_input = input("Press Enter to continue to the next David-run sample, or 'q' to quit: ")
            if user_input.lower() == 'q':
                logger.info("Process terminated by user during David-run processing.")
                break

if __name__ == "__main__":
    main()
