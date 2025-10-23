import sys
import os
import torch
import torchvision.transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn
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

# iPad 12.9-inch 6th generation camera parameters (estimated)
IPAD_FOCAL_LENGTH = 5.9  # mm
IPAD_SENSOR_WIDTH = 5.6  # mm
IPAD_SENSOR_HEIGHT = 4.2  # mm

# Keypoint indices for sideseam and waistline
SIDESEAM_INDICES = [11, 12, 13, 14]
WAISTLINE_INDICES = [14, 15, 16]

# Conversion factor: 1 meter = 39.3701 inches
METER_TO_INCHES = 39.3701

class DrHartCrisDataset(Dataset):
    def __init__(self, image_dirs, depth_dirs):
        self.data = []
        for image_dir, depth_dir in zip(image_dirs, depth_dirs):
            image_path = Path(image_dir)
            if not image_path.exists():
                print(f"Warning: Image directory not found: {image_path}")
                continue

            for img_file in image_path.glob('*.png'):
                depth_file = os.path.join(depth_dir, f'depth_{img_file.stem[-2:]}.npy')
                if os.path.exists(img_file) and os.path.exists(depth_file):
                    self.data.append((img_file, depth_file))
                else:
                    if not os.path.exists(img_file):
                        print(f"Warning: Image file not found: {img_file}")
                    if not os.path.exists(depth_file):
                        print(f"Warning: Depth file not found: {depth_file}")
            if not self.data:
                print("No data found. Please check your directories and data format.")

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
    # Load the base KeypointRCNN model
    model = keypointrcnn_resnet50_fpn(pretrained=False, num_classes=14, num_keypoints=39)

    # Check if a GPU is available, and if not, load the model on the CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the state dict from the pretrained DeepFashion2 checkpoint with the appropriate device
    state_dict = torch.load(checkpoint_path, map_location=device)  # <-- Add map_location to handle CPU environments

    # Adjust keypoint predictor layer
    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    num_keypoints = 39  # The number of keypoints used during training
    model.roi_heads.keypoint_predictor.kps_score_lowres = torch.nn.Conv2d(in_features, num_keypoints, kernel_size=1, stride=1)

    # Load the model state dict (ignore mismatches related to the keypoint predictor)
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


# Predict keypoints using the model
def predict_keypoints(model, image):
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract the predicted keypoints and labels
    keypoints = predictions[0]['keypoints']
    labels = predictions[0]['labels']

    # Filter keypoints specific to the "short sleeve top" (assuming category ID 1)
    short_sleeve_top_indices = (labels == 1).nonzero(as_tuple=True)[0]

    if len(short_sleeve_top_indices) > 0:
        # Get the first 25 keypoints for the short sleeve top
        short_sleeve_top_keypoints = keypoints[short_sleeve_top_indices[0]].cpu().numpy()[:25]
        return short_sleeve_top_keypoints
    else:
        print("No short sleeve top detected in the image.")
        return np.array([])

# Convert pixel-space keypoints to real-space coordinates
def pixel_to_real_space(keypoints, depth_map, image_width, image_height):
    # Use image dimensions for intrinsic parameters calculations
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

# Calculate the Euclidean distance between keypoints
def calculate_distance(keypoints):
    return np.sum(np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=1))

def process_sample(sample, keypoint_model):
    image = sample['image']
    depth_map = sample['depth_map']

    # Predict keypoints using the keypoint model
    predicted_keypoints = predict_keypoints(keypoint_model, image)

    # If no keypoints for short sleeve top, return empty result
    if len(predicted_keypoints) == 0:
        return None

    # Use actual image dimensions in processing
    image_width, image_height = image.size

    # Resize the depth map to match the size of the image
    depth_map_resized = cv2.resize(depth_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

    # Convert keypoints to real-world space based on the resized depth map
    real_space_keypoints = pixel_to_real_space(predicted_keypoints, depth_map_resized, image_width, image_height)

    # Calculate distances in meters
    sideseam_distance_meters = calculate_distance(real_space_keypoints[SIDESEAM_INDICES])
    waistline_distance_meters = calculate_distance(real_space_keypoints[WAISTLINE_INDICES])

    # Convert meters to inches
    sideseam_distance_inches = sideseam_distance_meters * METER_TO_INCHES
    waistline_distance_inches = waistline_distance_meters * METER_TO_INCHES

    return {
        'image': image,
        'predicted_keypoints': predicted_keypoints,  # Store the predicted keypoints
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
    
    # Ensure keypoints are properly handled; only 25 keypoints should be shown
    if keypoints is not None and len(keypoints) > 0:
        keypoints_np = np.array(keypoints)  # Only first 25 keypoints
        ax1.scatter(keypoints_np[:, 0], keypoints_np[:, 1], c='r', s=20)  # Plot keypoints
    ax1.set_title('Image with Predicted Keypoints')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(results['depth_map'], cmap='plasma')
    ax2.set_title('Depth Map')

    ax3d = fig.add_subplot(2, 3, 3, projection='3d')
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
    parser = argparse.ArgumentParser(description='Process garment images to calculate measurements using true depth and predicted keypoints.')
    parser.add_argument('--image_dirs', nargs='+', required=True, help='List of directories containing images.')
    parser.add_argument('--depth_dirs', nargs='+', required=True, help='List of directories containing depth maps.')
    parser.add_argument('--keypoint_model_path', required=True, help='Path to the keypoint model checkpoint.')
    args = parser.parse_args()

    # Load the custom-trained KeypointRCNN model
    keypoint_model = load_keypoint_model(args.keypoint_model_path)

    # Initialize dataset
    dataset = DrHartCrisDataset(args.image_dirs, args.depth_dirs)

    print("Processing...")
    for i in range(len(dataset)):
        sample = dataset[i]
        results = process_sample(sample, keypoint_model)
        if results:
            visualize_results(sample['image'], results['predicted_keypoints'], results)
        
        user_input = input("Press Enter to continue to the next sample, or 'q' to quit: ")
        if user_input.lower() == 'q':
            break

if __name__ == "__main__":
    main()