import sys
import os
import torch
import csv
import torchvision.transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.segmentation import deeplabv3_resnet101
import numpy as np
import logging
from PIL import Image
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

# Function to extract filenames from the CSV output of the first script
def get_processed_filenames(csv_filename):
    filenames = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_path = row['directory']
            filenames.append(image_path)
    return filenames

class DeepFashion2Dataset(Dataset):
    def __init__(self, image_paths, max_samples=None):
        self.data = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                self.data.append(img_path)
            else:
                logger.warning(f"Image file not found: {img_path}")

            if max_samples and len(self.data) >= max_samples:
                break

        logger.info(f"Loaded {len(self.data)} image samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            image = None
        return {
            'image': image,
            'img_path': img_path
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

def load_keypoint_model(checkpoint_path):
    model = keypointrcnn_resnet50_fpn(weights=None, num_classes=14, num_keypoints=39)
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

def load_segmentation_model():
    segmentation_model = deeplabv3_resnet101(weights='DEFAULT')
    segmentation_model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    segmentation_model.to(device)
    logger.info("Segmentation model loaded successfully.")
    return segmentation_model, device

def estimate_depth(model, image):
    image_np = np.array(image)
    with torch.no_grad():
        depth = model.infer_image(image_np)
    return depth

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

    person_class = 15  # COCO class index for 'person' in DeepLabV3
    binary_mask = (mask == person_class).astype(np.uint8)
    logger.debug("Segmentation mask generated.")
    return binary_mask

def predict_keypoints(model, image):
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        predictions = model(image_tensor)

    if not predictions or len(predictions) == 0:
        logger.warning("No predictions made by the keypoint model.")
        return np.array([])

    keypoints = predictions[0]['keypoints']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Select the prediction with the highest score
    if len(scores) > 0:
        best_idx = scores.argmax().item()
        best_keypoints = keypoints[best_idx].cpu().numpy()
        logger.debug("Keypoints predicted successfully.")
        return best_keypoints
    else:
        logger.warning("No keypoints detected with a valid score.")
        return np.array([])

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
        x, y, v = kp
        if v <= 0:
            real_space_keypoints.append([np.nan, np.nan, np.nan])
            continue
        x = np.clip(x, 0, image_width - 1)
        y = np.clip(y, 0, image_height - 1)
        z = depth_map[int(y), int(x)]
        real_x = (x - cx) * z / fx
        real_y = (y - cy) * z / fy
        real_space_keypoints.append([real_x, real_y, z])

    return np.array(real_space_keypoints)

def calculate_distance(keypoints):
    keypoints = keypoints[~np.isnan(keypoints).any(axis=1)]
    if keypoints.shape[0] < 2:
        return 0.0
    return np.sum(np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=1))

def process_sample(depth_model, keypoint_model, segmentation_model, seg_device, sample):
    try:
        image = sample['image']
        filename = sample['img_path']

        if image is None:
            logger.error(f"Image data is None for {filename}. Skipping.")
            return {'filename': filename, 'error': 'Image data is None'}

        logger.info(f"Processing file: {filename}")

        # Generate segmentation mask
        segmentation_mask = get_segmentation_mask(segmentation_model, seg_device, image)
        image_width, image_height = image.size
        segmentation_mask_resized = cv2.resize(segmentation_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

        # Apply the segmentation mask to the image
        image_np = np.array(image)
        image_np[segmentation_mask_resized == 0] = 0  # Set background pixels to zero (black)
        masked_image = Image.fromarray(image_np)

        # Predict keypoints using the keypoint model on the masked image
        predicted_keypoints = predict_keypoints(keypoint_model, masked_image)
        if len(predicted_keypoints) == 0:
            logger.warning(f"No keypoints detected for {filename}.")
            return {'filename': filename, 'error': 'No keypoints detected'}

        # Estimate depth map on the masked image
        depth_map = estimate_depth(depth_model, masked_image)
        depth_map_resized = cv2.resize(depth_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
        normalized_depth_map = normalize_depth_range(depth_map_resized)

        # Convert pixel space keypoints to real space
        real_space_keypoints = pixel_to_real_space(predicted_keypoints, normalized_depth_map, image_width, image_height)

        # Calculate distances
        left_sideseam_kps = real_space_keypoints[LEFT_SIDESEAM_INDICES]
        right_sideseam_kps = real_space_keypoints[RIGHT_SIDESEAM_INDICES]
        waistline_kps = real_space_keypoints[WAISTLINE_INDICES]

        left_sideseam_distance_meters = calculate_distance(left_sideseam_kps)
        right_sideseam_distance_meters = calculate_distance(right_sideseam_kps)
        waistline_distance_meters = calculate_distance(waistline_kps)

        left_sideseam_distance_inches = left_sideseam_distance_meters * METER_TO_INCHES
        right_sideseam_distance_inches = right_sideseam_distance_meters * METER_TO_INCHES
        waistline_distance_inches = waistline_distance_meters * METER_TO_INCHES

        average_sideseam_distance_inches = (left_sideseam_distance_inches + right_sideseam_distance_inches) / 2
        average_sideseam_distance_meters = (left_sideseam_distance_meters + right_sideseam_distance_meters) / 2

        logger.info(f"Sample {filename} processed successfully with keypoint refinement using segmentation mask.")
        return {
            'image': image,
            'predicted_keypoints': predicted_keypoints,
            'real_space_keypoints': real_space_keypoints,
            'depth_map': normalized_depth_map,
            'segmentation_mask': segmentation_mask_resized,
            'left_sideseam_distance_meters': left_sideseam_distance_meters,
            'right_sideseam_distance_meters': right_sideseam_distance_meters,
            'average_sideseam_distance_meters': average_sideseam_distance_meters,
            'waistline_distance_meters': waistline_distance_meters,
            'left_sideseam_distance_inches': left_sideseam_distance_inches,
            'right_sideseam_distance_inches': right_sideseam_distance_inches,
            'average_sideseam_distance_inches': average_sideseam_distance_inches,
            'waistline_distance_inches': waistline_distance_inches,
            'filename': filename,
            'error': None
        }
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return {'filename': filename, 'error': str(e)}

def write_results_to_csv(results, csv_filename):
    """Write results to CSV with the exact required format"""
    fieldnames = [
        'filename',
        'directory',  # Added directory field
        'left_sideseam_distance_inches',
        'right_sideseam_distance_inches',
        'average_sideseam_distance_inches',
        'waistline_distance_inches',
        'left_sideseam_distance_meters',
        'right_sideseam_distance_meters',
        'average_sideseam_distance_meters',
        'waistline_distance_meters'
    ]

    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'filename': os.path.basename(results['filename']),  # Just the filename
            'directory': results['filename'],  # Full path
            'left_sideseam_distance_inches': f"{results.get('left_sideseam_distance_inches', 0):.2f}",
            'right_sideseam_distance_inches': f"{results.get('right_sideseam_distance_inches', 0):.2f}",
            'average_sideseam_distance_inches': f"{results.get('average_sideseam_distance_inches', 0):.2f}",
            'waistline_distance_inches': f"{results.get('waistline_distance_inches', 0):.2f}",
            'left_sideseam_distance_meters': f"{results.get('left_sideseam_distance_meters', 0):.2f}",
            'right_sideseam_distance_meters': f"{results.get('right_sideseam_distance_meters', 0):.2f}",
            'average_sideseam_distance_meters': f"{results.get('average_sideseam_distance_meters', 0):.2f}",
            'waistline_distance_meters': f"{results.get('waistline_distance_meters', 0):.2f}"
        })

def main():
    parser = argparse.ArgumentParser(description='Process DeepFashion2 images to calculate measurements using predicted depth and keypoints.')
    parser.add_argument('--csv_input', required=True, help='Path to the input CSV file containing image paths.')
    parser.add_argument('--csv_output', required=True, help='Path to the output CSV file.')
    parser.add_argument('--depth_model_path', required=True, help='Path to the depth model checkpoint.')
    parser.add_argument('--keypoint_model_path', required=True, help='Path to the keypoint model checkpoint.')
    args = parser.parse_args()

    # Load processed filenames from CSV
    processed_filenames = get_processed_filenames(args.csv_input)

    # Load dataset (no need for JSON, only images)
    dataset = DeepFashion2Dataset(processed_filenames)

    # Check for loaded data
    if len(dataset) == 0:
        logger.error("No images found with the provided filenames.")
        return

    # Load models
    depth_model = load_depth_model(args.depth_model_path)
    keypoint_model = load_keypoint_model(args.keypoint_model_path)
    segmentation_model, seg_device = load_segmentation_model()

    # Processing loop
    for idx in tqdm(range(len(dataset)), desc="Processing Images"):
        sample = dataset[idx]
        print(f"\nProcessing image {idx + 1}/{len(dataset)}: {sample['img_path']}")

        results = process_sample(depth_model, keypoint_model, segmentation_model, seg_device, sample)
        if results.get('error') is None:
            write_results_to_csv(results, args.csv_output)
        else:
            logger.warning(f"Failed to process {results['filename']}: {results['error']}")

    logger.info(f"Results have been saved to {args.csv_output}")

if __name__ == "__main__":
    main()