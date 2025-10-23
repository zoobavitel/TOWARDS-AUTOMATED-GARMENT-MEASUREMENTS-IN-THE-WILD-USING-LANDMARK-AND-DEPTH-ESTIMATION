# Dr. Hart & Cris Thesis Code

This directory contains the four main Python scripts used for the garment measurement experiments in the thesis.

## Scripts

*   `pdpk_with_norms.py`: **Predicted Depth, Predicted Keypoints (with normalization)**
    *   This is the most advanced script, predicting both depth and keypoints from images.
    *   It uses a segmentation model to refine keypoints and normalizes the depth map.
    *   **To run:**
        ```bash
        python pdpk_with_norms.py --image_dirs path/to/image/dir1 path/to/image/dir2 --depth_model_path path/to/depth/model.pth --keypoint_model_path path/to/keypoint/model.pth
        ```

*   `pdtk.py`: **Predicted Depth, True Keypoints**
    *   This script uses ground truth keypoints from the dataset but predicts the depth.
    *   **To run:**
        ```bash
        python pdtk.py --anno_dirs path/to/anno/dir1 path/to/anno/dir2 --image_dirs path/to/image/dir1 path/to/image/dir2 --depth_model_path path/to/depth/model.pth
        ```

*   `tdpk.py`: **True Depth, Predicted Keypoints**
    *   This script uses ground truth depth from the dataset but predicts the keypoints.
    *   **To run:**
        ```bash
        python tdpk.py --image_dirs path/to/image/dir1 path/to/image/dir2 --depth_dirs path/to/depth/dir1 path/to/depth/dir2 --keypoint_model_path path/to/keypoint/model.pth
        ```

*   `tdtk.py`: **True Depth, True Keypoints**
    *   This script uses both ground truth depth and ground truth keypoints. It serves as a baseline for the other experiments.
    *   **To run:**
        ```bash
        python tdtk.py --anno_dirs path/to/anno/dir1 path/to/anno/dir2 --image_dirs path/to/image/dir1 path/to/image/dir2 --depth_dirs path/to/depth/dir1 path/to/depth/dir2
        ```

## Dependencies

*   PyTorch
*   TorchVision
*   Matplotlib
*   NumPy
*   Pillow
*   OpenCV
*   SciPy

The scripts also have a dependency on the [Depth-Anything-V2](https://github.com/LiheYoung/Depth-Anything) repository. It is assumed that this repository is cloned in the parent directory of this project.
