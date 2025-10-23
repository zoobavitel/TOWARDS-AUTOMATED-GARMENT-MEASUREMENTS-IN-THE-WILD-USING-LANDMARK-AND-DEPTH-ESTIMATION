import sys
import os
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
from PIL import Image
import json
import cv2
from torch.utils.data import Dataset
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# iPad 12.9-inch 6th generation camera parameters
IPAD_FOCAL_LENGTH = 5.9  # mm
IPAD_SENSOR_WIDTH = 5.6  # mm
IPAD_SENSOR_HEIGHT = 4.2  # mm

# Keypoint indices for sideseam and waistline
SIDESEAM_INDICES = [11, 12, 13, 14]
WAISTLINE_INDICES = [14, 15, 16]

# Conversion factor: 1 meter = 39.3701 inches
METER_TO_INCHES = 39.3701

class DrHartCrisDataset(Dataset):
    def __init__(self, anno_dirs, image_dirs, depth_dirs, category='short sleeve top'):
        self.data = []
        self.depth_dirs = depth_dirs
        for anno_dir, image_dir, depth_dir in zip(anno_dirs, image_dirs, depth_dirs):
            anno_path = Path(anno_dir)
            if not anno_path.exists():
                print(f"Warning: Annotations directory not found: {anno_path}")
                continue

            for anno_file in anno_path.glob('*.json'):
                with open(anno_file, 'r') as f:
                    anno = json.load(f)
                    item = anno.get('item1', {})
                    if item.get('category_name', '').lower() == category.lower():
                        img_file = anno_file.stem + '.png'
                        img_path = os.path.join(image_dir, img_file)

                        depth_file = os.path.join(depth_dir, f'depth_{anno_file.stem[-2:]}.npy')
                        
                        if os.path.exists(img_path) and os.path.exists(depth_file):
                            self.data.append((img_path, item, depth_file))
                        else:
                            if not os.path.exists(img_path):
                                print(f"Warning: Image file not found: {img_path}")
                            if not os.path.exists(depth_file):
                                print(f"Warning: Depth file not found: {depth_file}")
            if not self.data:
                print("No data found. Please check your directories and data format.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, anno, depth_file = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        keypoints = np.array(anno['landmarks'])
        depth_map = np.load(depth_file)

        return {
            'image': image,
            'keypoints': torch.tensor(keypoints).float(),
            'depth_map': depth_map
        }

def pixel_to_real_space(keypoints, depth_map, image_width, image_height):
    # Using actual image width and height dynamically
    fx = (IPAD_FOCAL_LENGTH / IPAD_SENSOR_WIDTH) * image_width
    fy = (IPAD_FOCAL_LENGTH / IPAD_SENSOR_HEIGHT) * image_height
    cx, cy = image_width / 2, image_height / 2

    depth_height, depth_width = depth_map.shape
    scale_x = depth_width / image_width
    scale_y = depth_height / image_height

    real_space_keypoints = []
    for kp in keypoints:
        x, y, _ = kp
        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        scaled_x = np.clip(scaled_x, 0, depth_width - 1)
        scaled_y = np.clip(scaled_y, 0, depth_height - 1)
        z = depth_map[scaled_y, scaled_x]
        real_x = (x - cx) * z / fx
        real_y = (y - cy) * z / fy
        real_space_keypoints.append([real_x, real_y, z])

    return np.array(real_space_keypoints)

def calculate_distance(keypoints):
    return np.sum(np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=1))

def process_sample(sample):
    image = sample['image']
    keypoints = sample['keypoints']
    depth_map = sample['depth_map']

    # Resize the depth map to match the size of the image
    depth_map_resized = cv2.resize(depth_map, image.size, interpolation=cv2.INTER_LINEAR)

    # Convert keypoints to real-world space based on the resized depth map
    real_space_keypoints = pixel_to_real_space(keypoints.numpy(), depth_map_resized, *image.size)

    # Calculate distances in meters
    sideseam_distance_meters = calculate_distance(real_space_keypoints[SIDESEAM_INDICES])
    waistline_distance_meters = calculate_distance(real_space_keypoints[WAISTLINE_INDICES])

    # Convert meters to inches
    sideseam_distance_inches = sideseam_distance_meters * METER_TO_INCHES
    waistline_distance_inches = waistline_distance_meters * METER_TO_INCHES

    return {
        'image': image,
        'depth_map': depth_map_resized,
        'real_space_keypoints': real_space_keypoints,
        'sideseam_distance_meters': sideseam_distance_meters,
        'waistline_distance_meters': waistline_distance_meters,
        'sideseam_distance_inches': sideseam_distance_inches,
        'waistline_distance_inches': waistline_distance_inches
    }

def visualize_results(image, keypoints, results):
    fig = plt.figure(figsize=(20, 15))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(np.array(image))
    keypoints_np = keypoints.cpu().numpy()
    ax1.scatter(keypoints_np[:, 0], keypoints_np[:, 1], c='r', s=20)
    
    # Connect Sideseam Keypoints
    sideseam_pts = keypoints_np[SIDESEAM_INDICES]
    ax1.plot(sideseam_pts[:, 0], sideseam_pts[:, 1], 'b-', lw=2, label='Sideseam')
    
    # Connect Waistline Keypoints
    waistline_pts = keypoints_np[WAISTLINE_INDICES]
    ax1.plot(waistline_pts[:, 0], waistline_pts[:, 1], 'g-', lw=2, label='Waistline')
    
    ax1.legend()
    ax1.set_title('Image with Keypoints and Connections')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(results['depth_map'], cmap='plasma')
    ax2.set_title('Depth Map')

    ax3d = fig.add_subplot(2, 3, 5, projection='3d')
    real_space_keypoints = results['real_space_keypoints']
    ax3d.scatter(real_space_keypoints[:, 0], real_space_keypoints[:, 1], real_space_keypoints[:, 2], c='r', s=20)
    ax3d.set_title('3D Keypoints')

    plt.tight_layout()
    plt.show()

    # Print distances in both meters and inches
    print(f"Side Seam Distance: {results['sideseam_distance_meters']:.4f} meters ({results['sideseam_distance_inches']:.2f} inches)")
    print(f"Waistline Distance: {results['waistline_distance_meters']:.4f} meters ({results['waistline_distance_inches']:.2f} inches)")

import argparse

def main():
    parser = argparse.ArgumentParser(description='Process garment images to calculate measurements using true depth and true keypoints.')
    parser.add_argument('--anno_dirs', nargs='+', required=True, help='List of directories containing annotations.')
    parser.add_argument('--image_dirs', nargs='+', required=True, help='List of directories containing images.')
    parser.add_argument('--depth_dirs', nargs='+', required=True, help='List of directories containing depth maps.')
    args = parser.parse_args()

    # Load dataset
    dataset = DrHartCrisDataset(args.anno_dirs, args.image_dirs, args.depth_dirs)
    print("Processing...")
    for i in range(len(dataset)):
        sample = dataset[i]
        results = process_sample(sample)
        visualize_results(sample['image'], sample['keypoints'], results)
        user_input = input("Press Enter to continue to the next sample, or 'q' to quit: ")
        if user_input.lower() == 'q':
            break

if __name__ == "__main__":
    main()

        
