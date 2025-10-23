import sys
import os
import torch
import csv
import matplotlib.pyplot as plt
import numpy as np
import logging
from PIL import Image
import json
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import warnings
from tqdm import tqdm

import argparse

# Suppress specific warnings if necessary
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# iPhone 6 camera parameters
IPHONE6_FOCAL_LENGTH = 4.15  # mm
IPHONE6_SENSOR_WIDTH = 4.8  # mm
IPHONE6_SENSOR_HEIGHT = 3.6  # mm

# Keypoint indices for sideseam and waistline
LEFT_SIDESEAM_INDICES = [16, 17, 18, 19]
RIGHT_SIDESEAM_INDICES = [11, 12, 13, 14]
WAISTLINE_INDICES = [14, 15, 16]

# Conversion factor: 1 meter = 39.3701 inches
METER_TO_INCHES = 39.3701

# Import DepthAnythingV2
depth_anything_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Depth-Anything-V2'))
sys.path.append(depth_anything_path)
sys.path.append(os.path.join(depth_anything_path, 'metric_depth'))

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    raise ImportError("Could not import DepthAnythingV2. Please check your directory structure and Python path.")

class DeepFashion2Dataset(Dataset):
    def __init__(self, anno_dirs, image_dirs, category='short sleeve top', max_samples=None):
        self.data = []
        required_indices = set(LEFT_SIDESEAM_INDICES + RIGHT_SIDESEAM_INDICES + WAISTLINE_INDICES)
        
        for anno_dir, image_dir in zip(anno_dirs, image_dirs):
            anno_path = Path(anno_dir)
            if not anno_path.exists():
                logger.warning(f"Annotations directory not found: {anno_path}")
                continue

            for anno_file in anno_path.glob('*.json'):
                with open(anno_file, 'r') as f:
                    anno = json.load(f)
                    for item_key, item in anno.items():
                        if isinstance(item, dict) and item.get('category_name', '').lower() == category.lower():
                            landmarks = np.array(item['landmarks']).reshape(-1, 3)
                            segmentation = item.get('segmentation', [])
                            
                            # Check visibility only for required keypoints
                            required_landmarks = landmarks[list(required_indices)]
                            visible_required_landmarks = required_landmarks[required_landmarks[:, 2] > 0]

                            # Only consider samples where all required keypoints are visible
                            if len(visible_required_landmarks) == len(required_landmarks):
                                img_file = anno_file.stem + '.jpg'
                                img_path = os.path.join(image_dir, img_file)
                                if os.path.exists(img_path):
                                    self.data.append((img_path, landmarks, segmentation, anno_file.name))
                                else:
                                    logger.warning(f"Image file not found: {img_path}")
                            else:
                                logger.info(f"Skipping {anno_file.stem}: Not all required keypoints are visible")
                
                if max_samples and len(self.data) >= max_samples:
                    break
            if max_samples and len(self.data) >= max_samples:
                break
        
        logger.info(f"Loaded {len(self.data)} samples with all required visible keypoints")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, keypoints, segmentation, anno_filename = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        return {
            'image': image,
            'keypoints': torch.tensor(keypoints).float(),
            'segmentation': segmentation,
            'img_path': img_path,
            'original_size': image.size,
            'anno_filename': anno_filename
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

def normalize_depth_range(depth_map, target_min=0.5, target_max=3.0):
    current_min, current_max = np.min(depth_map), np.max(depth_map)
    normalized_depth = (depth_map - current_min) / (current_max - current_min)
    scaled_depth = normalized_depth * (target_max - target_min) + target_min
    return scaled_depth

def pixel_to_real_space(keypoints, depth_map, image_width, image_height):
    fx = (IPHONE6_FOCAL_LENGTH / IPHONE6_SENSOR_WIDTH) * image_width
    fy = (IPHONE6_FOCAL_LENGTH / IPHONE6_SENSOR_HEIGHT) * image_height
    cx, cy = image_width / 2, image_height / 2

    real_space_keypoints = []
    for kp in keypoints:
        x, y, _ = kp
        z = depth_map[int(y), int(x)]
        real_x = (x - cx) * z / fx
        real_y = (y - cy) * z / fy
        real_space_keypoints.append([real_x, real_y, z])

    return np.array(real_space_keypoints)

def calculate_distance(keypoints):
    if keypoints.shape[0] < 2:
        return 0.0
    return np.sum(np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=1))

def process_sample(depth_model, sample):
    image = sample['image']
    keypoints = sample['keypoints']
    segmentation = sample['segmentation']
    filename = sample['anno_filename']
    img_path = sample['img_path']

    logger.info(f"Processing file: {filename}")

    # Check if there are valid keypoints
    if keypoints is None or len(keypoints) == 0:
        logger.warning(f"No valid keypoints found for {filename}. Skipping.")
        return None

    # Get depth map
    depth_map = estimate_depth(depth_model, image)
    image_width, image_height = image.size
    depth_map_resized = cv2.resize(depth_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    normalized_depth_map = normalize_depth_range(depth_map_resized)

    # Create segmentation mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for seg in segmentation:
        seg_array = np.array(seg).reshape(-1, 2)
        cv2.fillPoly(mask, [seg_array.astype(int)], 1)

    # Convert pixel space keypoints to real space
    real_space_keypoints = pixel_to_real_space(keypoints.numpy(), normalized_depth_map, image_width, image_height)

    # Calculate distances
    left_sideseam_distance_meters = calculate_distance(real_space_keypoints[LEFT_SIDESEAM_INDICES])
    right_sideseam_distance_meters = calculate_distance(real_space_keypoints[RIGHT_SIDESEAM_INDICES])
    waistline_distance_meters = calculate_distance(real_space_keypoints[WAISTLINE_INDICES])

    left_sideseam_distance_inches = left_sideseam_distance_meters * METER_TO_INCHES
    right_sideseam_distance_inches = right_sideseam_distance_meters * METER_TO_INCHES
    waistline_distance_inches = waistline_distance_meters * METER_TO_INCHES

    average_sideseam_distance_inches = (left_sideseam_distance_inches + right_sideseam_distance_inches) / 2

    logger.info(f"Sample {filename} processed successfully.")
    return {
        'image': image,
        'keypoints': keypoints.numpy(),
        'real_space_keypoints': real_space_keypoints,
        'depth_map': normalized_depth_map,
        'segmentation_mask': mask,
        'left_sideseam_distance_meters': left_sideseam_distance_meters,
        'right_sideseam_distance_meters': right_sideseam_distance_meters,
        'waistline_distance_meters': waistline_distance_meters,
        'left_sideseam_distance_inches': left_sideseam_distance_inches,
        'right_sideseam_distance_inches': right_sideseam_distance_inches,
        'average_sideseam_distance_inches': average_sideseam_distance_inches,
        'waistline_distance_inches': waistline_distance_inches,
        'filename': filename,
        'img_path': img_path
    }

def write_results_to_csv(results, csv_filename):
    fieldnames = [
        'filename', 
        'img_path',
        'left_sideseam_distance_inches', 
        'right_sideseam_distance_inches', 
        'average_sideseam_distance_inches', 
        'waistline_distance_inches'
    ]
    
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'filename': results['filename'],
            'img_path': results['img_path'],
            'left_sideseam_distance_inches': f"{results['left_sideseam_distance_inches']:.2f}",
            'right_sideseam_distance_inches': f"{results['right_sideseam_distance_inches']:.2f}",
            'average_sideseam_distance_inches': f"{results['average_sideseam_distance_inches']:.2f}",
            'waistline_distance_inches': f"{results['waistline_distance_inches']:.2f}"
        })

def main():
    parser = argparse.ArgumentParser(description='Process DeepFashion2 images to calculate measurements using predicted depth and true keypoints.')
    parser.add_argument('--anno_dirs', nargs='+', required=True, help='List of directories containing annotations.')
    parser.add_argument('--image_dirs', nargs='+', required=True, help='List of directories containing images.')
    parser.add_argument('--depth_model_path', required=True, help='Path to the depth model checkpoint.')
    parser.add_argument('--csv_output', required=True, help='Path to the output CSV file.')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process.')
    args = parser.parse_args()

    # Check that all directories exist
    for dir_path in args.anno_dirs + args.image_dirs:
        if not os.path.exists(dir_path):
            logger.error(f"Directory does not exist: {dir_path}")
            return

    dataset = DeepFashion2Dataset(args.anno_dirs, args.image_dirs, max_samples=args.max_samples)
    
    # Ensure that there is at least one valid sample
    if len(dataset) == 0:
        logger.error("No valid samples found with all visible keypoints.")
        return

    # Load depth model
    depth_model = load_depth_model(args.depth_model_path)

    logger.info("Processing samples...")
    for idx in tqdm(range(len(dataset)), desc="Processing Samples"):
        sample = dataset[idx]
        print(f"\nProcessing sample {idx + 1}/{len(dataset)}: {sample['anno_filename']}")
        try:
            results = process_sample(depth_model, sample)
            if results is not None:
                # Optionally visualize results (commented out for batch processing)
                # visualize_results(sample['image'], results)
                write_results_to_csv(results, args.csv_output)
        except Exception as e:
            logger.error(f"Error processing sample {sample['anno_filename']}: {e}", exc_info=True)
            continue  # Skip to the next sample

    logger.info(f"Results have been saved to {args.csv_output}")

if __name__ == "__main__":
    main()

