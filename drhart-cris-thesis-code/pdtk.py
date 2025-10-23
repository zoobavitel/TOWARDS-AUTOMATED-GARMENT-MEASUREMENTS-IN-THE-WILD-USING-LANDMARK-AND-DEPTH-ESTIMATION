import sys
import os
import argparse
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

# Add both possible paths to the Python path
depth_anything_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Depth-Anything-V2'))
sys.path.append(depth_anything_path)
sys.path.append(os.path.join(depth_anything_path, 'metric_depth'))

# Try to import DepthAnythingV2 from both possible locations
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    try:
        from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        raise ImportError("Could not import DepthAnythingV2. Please check your directory structure and Python path.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

# iPad 12.9-inch 6th generation camera parameters (estimated)
IPAD_FOCAL_LENGTH = 5.9  # mm
IPAD_SENSOR_WIDTH = 5.6  # mm
IPAD_SENSOR_HEIGHT = 4.2  # mm

# Keypoint indices (adjusted for sideseam and waistline)
SIDESEAM_INDICES = [11, 12, 13, 14]
WAISTLINE_INDICES = [14, 15, 16]

class DrHartCrisDataset(Dataset):
    def __init__(self, anno_dirs, image_dirs, category='short sleeve top'):
        self.data = []
        for anno_dir, image_dir in zip(anno_dirs, image_dirs):
            anno_dir_path = Path(anno_dir)
            if not anno_dir_path.exists():
                print(f"Warning: Annotations directory not found: {anno_dir_path}")
                continue

            for anno_file in anno_dir_path.glob('*.json'):
                with open(anno_file, 'r') as f:
                    anno = json.load(f)
                    item = anno.get('item1', {})
                    if item.get('category_name', '').lower() == category.lower():
                        img_file = anno_file.stem + '.png'
                        img_path = os.path.join(image_dir, img_file)
                        if os.path.exists(img_path):
                            self.data.append((img_path, item))
                        else:
                            print(f"Warning: Image file not found: {img_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, anno = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        try:
            keypoints = np.array(anno['landmarks'])
        except KeyError as e:
            raise KeyError(f"Missing key {e} in annotation for image {img_path}")

        return {
            'image': image,
            'keypoints': torch.tensor(keypoints).float(),
            'original_size': image.size
        }

def load_depth_model(model_path):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthAnythingV2()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    model.to(DEVICE)  # Move model to the same device
    model.eval()
    return model

def estimate_depth(model, image):
    # Convert PIL Image to numpy array if it isn't already
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Get the depth map directly from the model
    depth = model.infer_image(image_np)

    print(f"Depth map shape: {depth.shape}")  # Should match (height, width)

    return depth

def pixel_to_real_space(keypoints, depth_map):
    depth_height, depth_width = depth_map.shape

    # Use actual image dimensions from depth map
    fx = (IPAD_FOCAL_LENGTH / IPAD_SENSOR_WIDTH) * depth_width
    fy = (IPAD_FOCAL_LENGTH / IPAD_SENSOR_HEIGHT) * depth_height
    cx, cy = depth_width / 2, depth_height / 2

    real_space_keypoints = []
    for kp in keypoints:
        x, y, _ = kp
        x = np.clip(int(x), 0, depth_width - 1)
        y = np.clip(int(y), 0, depth_height - 1)
        z = depth_map[y, x]

        if z > 0:
            real_x = (x - cx) * z / fx
            real_y = (y - cy) * z / fy
        else:
            real_x, real_y, z = np.nan, np.nan, np.nan  # Handle invalid depth

        real_space_keypoints.append([real_x, real_y, z])

    return np.array(real_space_keypoints)

def calculate_distance(keypoints):
    # Remove keypoints with NaN values
    valid_keypoints = keypoints[~np.isnan(keypoints).any(axis=1)]
    if len(valid_keypoints) < 2:
        return np.nan  # Not enough valid points to calculate distance
    # Sum distances between consecutive keypoints
    distances = np.linalg.norm(valid_keypoints[1:] - valid_keypoints[:-1], axis=1)
    total_distance = np.sum(distances)
    return total_distance

def process_sample(depth_model, sample):
    image = sample['image']
    keypoints = sample['keypoints']

    # Estimate depth using the depth model
    depth_map = estimate_depth(depth_model, image)

    # Convert keypoints to real-world space
    real_space_keypoints = pixel_to_real_space(keypoints.numpy(), depth_map)

    sideseam_distance = calculate_distance(real_space_keypoints[SIDESEAM_INDICES])
    waistline_distance = calculate_distance(real_space_keypoints[WAISTLINE_INDICES])

    return {
        'image': image,
        'depth_map': depth_map,
        'real_space_keypoints': real_space_keypoints,
        'sideseam_distance': sideseam_distance,
        'waistline_distance': waistline_distance
    }

def visualize_results(image, keypoints, results):
    fig = plt.figure(figsize=(20, 15))

    # Image with keypoints
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(np.array(image))
    keypoints_np = keypoints.cpu().numpy()
    ax1.scatter(keypoints_np[:, 0], keypoints_np[:, 1], c='r', s=20)

    # Draw Sideseam and Waistline connections
    sideseam_pts = keypoints_np[SIDESEAM_INDICES]
    ax1.plot(sideseam_pts[:, 0], sideseam_pts[:, 1], 'b-', lw=2, label='Sideseam')

    waistline_pts = keypoints_np[WAISTLINE_INDICES]
    ax1.plot(waistline_pts[:, 0], waistline_pts[:, 1], 'g-', lw=2, label='Waistline')

    ax1.legend()
    ax1.set_title('Image with Keypoints')

    # Depth map visualization
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(results['depth_map'], cmap='plasma')
    ax2.set_title('Depth Map')

    # 3D keypoints
    ax3d = fig.add_subplot(2, 3, 3, projection='3d')
    real_space_keypoints = results['real_space_keypoints']
    valid_keypoints = real_space_keypoints[~np.isnan(real_space_keypoints).any(axis=1)]
    if len(valid_keypoints) > 0:
        ax3d.scatter(valid_keypoints[:, 0], valid_keypoints[:, 1], valid_keypoints[:, 2], c='r', s=20)
        ax3d.set_xlabel('X (meters)')
        ax3d.set_ylabel('Y (meters)')
        ax3d.set_zlabel('Z (meters)')
    ax3d.set_title('3D Keypoints')

    plt.tight_layout()
    plt.show()

    # Print distances in both meters
    print(f"Side Seam Distance: {results['sideseam_distance']:.4f} meters")
    print(f"Waistline Distance: {results['waistline_distance']:.4f} meters")

def main():
    parser = argparse.ArgumentParser(description='Process garment images to calculate measurements using true keypoints and predicted depth.')
    parser.add_argument('--anno_dirs', nargs='+', required=True, help='List of directories containing annotations.')
    parser.add_argument('--image_dirs', nargs='+', required=True, help='List of directories containing images.')
    parser.add_argument('--depth_model_path', required=True, help='Path to the depth model checkpoint.')
    args = parser.parse_args()

    # Check that all directories exist
    for dir_path in args.anno_dirs + args.image_dirs:
        if not os.path.exists(dir_path):
            print(f"Error: Directory does not exist: {dir_path}")
            return

    # Load dataset
    dataset = DrHartCrisDataset(args.anno_dirs, args.image_dirs)
    if len(dataset) == 0:
        print("No data found. Please check your annotation and image directories.")
        return

    # Load depth model
    depth_model = load_depth_model(args.depth_model_path)

    try:
        for i in range(len(dataset)):
            # Fetch the sample
            sample = dataset[i]

            # Process the sample to get depth map and measurements
            results = process_sample(depth_model, sample)

            # Visualize results (keypoints, depth map, and 3D keypoints)
            visualize_results(sample['image'], sample['keypoints'], results)

            # Handle user input to continue or stop
            user_input = input("Press Enter to continue to the next sample, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
