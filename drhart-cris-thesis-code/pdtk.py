import sys
import os
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import matplotlib.pyplot as plt
import numpy as np
import logging
from PIL import Image
import json
import cv2
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import map_coordinates

# Import DepthAnythingV2
depth_anything_paths = [
    "C:/Users/crisz/Desktop/Prod-Thesis/Production/Depth-Anything-V2/metric_depth",
    "C:/Users/crisz/Desktop/Prod-Thesis/Production/Depth-Anything-V2"
]
for path in depth_anything_paths:
    sys.path.append(os.path.abspath(path))

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    raise ImportError("Could not import DepthAnythingV2. Please check your directory structure and Python path.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IPAD_FOCAL_LENGTH = 5.9  # mm
IPAD_SENSOR_WIDTH = 5.6  # mm
IPAD_SENSOR_HEIGHT = 4.2  # mm
LEFT_SIDESEAM_INDICES = [16, 17, 18, 19]
RIGHT_SIDESEAM_INDICES = [11, 12, 13, 14]
WAISTLINE_INDICES = [14, 15, 16]
METER_TO_INCHES = 39.3701

class DrHartCrisDataset(Dataset):
    def __init__(self, anno_dirs, image_dirs, category='short sleeve top'):
        self.data = []
        for anno_dir, image_dir in zip(anno_dirs, image_dirs):
            anno_path = Path(anno_dir)
            if not anno_path.exists():
                logger.warning(f"Annotations directory not found: {anno_path}")
                continue

            for anno_file in anno_path.glob('*.json'):
                with open(anno_file, 'r') as f:
                    anno = json.load(f)
                    item = anno.get('item1', {})
                    if item.get('category_name', '').lower() == category.lower():
                        img_file = anno_file.stem + '.png'
                        img_path = os.path.join(image_dir, img_file)
                        if os.path.exists(img_path):
                            self.data.append((img_path, item))
                        else:
                            logger.warning(f"Image file not found: {img_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, anno = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        keypoints = np.array(anno['landmarks']) if 'landmarks' in anno else None
        return {
            'image': image,
            'keypoints': torch.tensor(keypoints).float() if keypoints is not None else None,
            'category': anno['category_name'],
            'original_size': image.size
        }

def load_depth_model(model_path):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthAnythingV2()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    logger.info("DepthAnythingV2 model loaded successfully.")
    return model

def estimate_depth(model, image):
    image_np = np.array(image)
    with torch.no_grad():
        depth = model.infer_image(image_np)
    return depth

def load_segmentation_model():
    segmentation_model = deeplabv3_resnet101(weights='DEFAULT')
    segmentation_model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    segmentation_model.to(device)
    logger.info("Segmentation model loaded successfully.")
    return segmentation_model, device

def get_segmentation_mask(model, device, image):
    preprocess = T.Compose([
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = torch.argmax(output, dim=0).cpu().numpy()
    person_class = 15
    binary_mask = (mask == person_class).astype(np.uint8)
    return binary_mask

def get_refined_subpixel_depth(x, y, depth_map, window_size=5):
    h, w = depth_map.shape
    x1, x2 = max(0, int(x) - window_size), min(w, int(x) + window_size + 1)
    y1, y2 = max(0, int(y) - window_size), min(h, int(y) + window_size + 1)
    local_depth = depth_map[y1:y2, x1:x2]
    local_median = np.median(local_depth)
    depth = map_coordinates(depth_map, [[y], [x]], order=1, mode='nearest')[0]
    
    if abs(depth - local_median) > 0.1 * local_median:  # 10% threshold
        return local_median
    return depth

def pixel_to_real_space(keypoints, depth_map, image_width, image_height):
    fx = (IPAD_FOCAL_LENGTH / IPAD_SENSOR_WIDTH) * image_width
    fy = (IPAD_FOCAL_LENGTH / IPAD_SENSOR_HEIGHT) * image_height
    cx, cy = image_width / 2, image_height / 2
    depth_height, depth_width = depth_map.shape
    scale_x = depth_width / image_width
    scale_y = depth_height / image_height

    real_space_keypoints = []
    for kp in keypoints:
        x, y, _ = kp
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        z = get_refined_subpixel_depth(scaled_x, scaled_y, depth_map)
        real_x = (x - cx) * z / fx
        real_y = (y - cy) * z / fy
        real_space_keypoints.append([real_x, real_y, z])

    return np.array(real_space_keypoints)

def normalize_depth_range(depth_map, target_min=0.5, target_max=3.0):
    current_min, current_max = np.min(depth_map), np.max(depth_map)
    normalized_depth = (depth_map - current_min) / (current_max - current_min)
    scaled_depth = normalized_depth * (target_max - target_min) + target_min
    return scaled_depth

def enforce_depth_consistency(keypoints, max_depth_diff=0.1):
    adjusted_keypoints = keypoints.copy()
    median_depth = np.median(keypoints[:, 2])
    
    for idx in range(len(keypoints)):
        if abs(adjusted_keypoints[idx, 2] - median_depth) > max_depth_diff:
            adjusted_keypoints[idx, 2] = median_depth
    
    return adjusted_keypoints

def validate_keypoints_with_segmentation(keypoints, segmentation_mask):
    valid_keypoints = []
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if 0 <= x < segmentation_mask.shape[1] and 0 <= y < segmentation_mask.shape[0]:
            if segmentation_mask[y, x] > 0:
                valid_keypoints.append(True)
            else:
                valid_keypoints.append(False)
        else:
            valid_keypoints.append(False)
    return np.array(valid_keypoints)

def calculate_distance(keypoints):
    if keypoints.shape[0] < 2:
        return 0.0
    return np.sum(np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=1))

def process_sample(depth_model, segmentation_model, seg_device, sample):
    image = sample['image']
    keypoints = sample['keypoints']

    if keypoints is None:
        logger.warning("No keypoints available in the sample. Skipping.")
        return None

    depth_map = estimate_depth(depth_model, image)
    segmentation_mask = get_segmentation_mask(segmentation_model, seg_device, image)

    image_width, image_height = image.size
    segmentation_mask_resized = cv2.resize(segmentation_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    depth_map_resized = cv2.resize(depth_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

    normalized_depth_map = normalize_depth_range(depth_map_resized)
    
    real_space_keypoints = pixel_to_real_space(keypoints.numpy(), normalized_depth_map, image_width, image_height)
    
    final_keypoints = enforce_depth_consistency(real_space_keypoints)

    valid_keypoints = validate_keypoints_with_segmentation(keypoints.numpy(), segmentation_mask_resized)

    left_sideseam_distance_meters = calculate_distance(final_keypoints[LEFT_SIDESEAM_INDICES])
    right_sideseam_distance_meters = calculate_distance(final_keypoints[RIGHT_SIDESEAM_INDICES])
    waistline_distance_meters = calculate_distance(final_keypoints[WAISTLINE_INDICES])

    left_sideseam_distance_inches = left_sideseam_distance_meters * METER_TO_INCHES
    right_sideseam_distance_inches = right_sideseam_distance_meters * METER_TO_INCHES
    waistline_distance_inches = waistline_distance_meters * METER_TO_INCHES

    logger.info("Sample processed successfully with depth refinement.")
    return {
        'image': image,
        'keypoints': keypoints,
        'real_space_keypoints': final_keypoints,
        'depth_map': normalized_depth_map,
        'segmentation_mask': segmentation_mask_resized,
        'valid_keypoints': valid_keypoints,
        'left_sideseam_distance_meters': left_sideseam_distance_meters,
        'right_sideseam_distance_meters': right_sideseam_distance_meters,
        'waistline_distance_meters': waistline_distance_meters,
        'left_sideseam_distance_inches': left_sideseam_distance_inches,
        'right_sideseam_distance_inches': right_sideseam_distance_inches,
        'waistline_distance_inches': waistline_distance_inches
    }

def visualize_results(image, keypoints, results):
    fig = plt.figure(figsize=(20, 15))

    # Plot the original image with keypoints
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(np.array(image))
    valid_keypoints = results['valid_keypoints']
    pixel_keypoints = keypoints.numpy()[:, :2]
    ax1.scatter(pixel_keypoints[valid_keypoints, 0], pixel_keypoints[valid_keypoints, 1], c='g', s=20, label='Valid Keypoints')
    ax1.scatter(pixel_keypoints[~valid_keypoints, 0], pixel_keypoints[~valid_keypoints, 1], c='r', s=20, label='Invalid Keypoints')
    
    # Add labels for each keypoint
    for i, (x, y) in enumerate(pixel_keypoints):
        ax1.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                     fontsize=8, color='white', backgroundcolor='black')
    
    ax1.set_title('Image with Keypoints')
    ax1.legend()

    # Plot lines for Sideseam and Waistline keypoints
    left_sideseam = pixel_keypoints[LEFT_SIDESEAM_INDICES]
    right_sideseam = pixel_keypoints[RIGHT_SIDESEAM_INDICES]
    waistline = pixel_keypoints[WAISTLINE_INDICES]
    
    ax1.plot(left_sideseam[:, 0], left_sideseam[:, 1], 'b-', lw=2, label='Left Sideseam')
    ax1.plot(right_sideseam[:, 0], right_sideseam[:, 1], 'g-', lw=2, label='Right Sideseam')
    ax1.plot(waistline[:, 0], waistline[:, 1], 'r-', lw=2, label='Waistline')

    # Depth map visualization
    ax2 = fig.add_subplot(2, 3, 2)
    depth_map = results['depth_map']
    im = ax2.imshow(depth_map, cmap='plasma')
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
    left_sideseam_3d = real_space_keypoints[LEFT_SIDESEAM_INDICES]
    right_sideseam_3d = real_space_keypoints[RIGHT_SIDESEAM_INDICES]
    waistline_3d = real_space_keypoints[WAISTLINE_INDICES]
    
    ax4.plot(left_sideseam_3d[:, 0], left_sideseam_3d[:, 1], left_sideseam_3d[:, 2], 'b-', linewidth=2, label='Left Sideseam')
    ax4.plot(right_sideseam_3d[:, 0], right_sideseam_3d[:, 1], right_sideseam_3d[:, 2], 'g-', linewidth=2, label='Right Sideseam')
    ax4.plot(waistline_3d[:, 0], waistline_3d[:, 1], waistline_3d[:, 2], 'r-', linewidth=2, label='Waistline')
    
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
    print(f"Left Sideseam Distance: {results['left_sideseam_distance_meters']:.4f} meters ({results['left_sideseam_distance_inches']:.2f} inches)")
    print(f"Right Sideseam Distance: {results['right_sideseam_distance_meters']:.4f} meters ({results['right_sideseam_distance_inches']:.2f} inches)")
    print(f"Waistline Distance: {results['waistline_distance_meters']:.4f} meters ({results['waistline_distance_inches']:.2f} inches)")

    # Print depths for each keypoint
    print("\nKeypoint Depths:")
    for i, kp in enumerate(real_space_keypoints):
        print(f"Keypoint {i}: {kp[2]:.4f} meters")

    # Print information about invalid keypoints
    invalid_count = np.sum(~valid_keypoints)
    print(f"\nInvalid Keypoints: {invalid_count}")
    if invalid_count > 0:
        invalid_indices = np.where(~valid_keypoints)[0]
        print("Invalid Keypoint Indices:", invalid_indices)

def main():
    anno_dirs = [
        r"C:\Users\crisz\Desktop\Prod-Thesis\DrHart&CrisData\cris-run\lidar\run1\annos",
        r"C:\Users\crisz\Desktop\Prod-Thesis\DrHart&CrisData\david-run\lidar\run1\annos"
    ]
    image_dirs = [
        r"C:\Users\crisz\Desktop\Prod-Thesis\DrHart&CrisData\cris-run\lidar\run1\image",
        r"C:\Users\crisz\Desktop\Prod-Thesis\DrHart&CrisData\david-run\lidar\run1\image"
    ]
    depth_model_path = r"C:\Users\crisz\Desktop\Prod-Thesis\Production\checkpoints\depth_anything_v2_metric_hypersim_vitl.pth"

    # Check that all directories exist
    for dir_path in anno_dirs + image_dirs:
        if not os.path.exists(dir_path):
            logger.error(f"Directory does not exist: {dir_path}")
            return

    dataset = DrHartCrisDataset(anno_dirs, image_dirs)
    if len(dataset) == 0:
        logger.error("No data found. Please check your annotation and image directories.")
        return

    # Load models
    depth_model = load_depth_model(depth_model_path)
    segmentation_model, seg_device = load_segmentation_model()

    logger.info("Processing samples...")
    for idx in range(len(dataset)):
        sample = dataset[idx]
        results = process_sample(depth_model, segmentation_model, seg_device, sample)
        
        if results is not None:
            visualize_results(sample['image'], sample['keypoints'], results)

        user_input = input("Press Enter to continue to the next sample, or 'q' to quit: ")
        if user_input.lower() == 'q':
            logger.info("Process terminated by user.")
            break

if __name__ == "__main__":
    main()