# DeepFashion2 Code

This directory contains the Python scripts for running the garment measurement experiments on the DeepFashion2 dataset.

## Scripts

*   `pdpk.py`: **Predicted Depth, Predicted Keypoints**
    *   This script predicts both depth and keypoints from images in the DeepFashion2 dataset.
    *   It takes a CSV file as input, which contains the paths to the images to be processed.
    *   **To run:**
        ```bash
        python pdpk.py --csv_input path/to/your/images.csv --csv_output path/to/your/results.csv --depth_model_path path/to/depth/model.pth --keypoint_model_path path/to/keypoint/model.pth
        ```

*   `pdtk.py`: **Predicted Depth, True Keypoints**
    *   This script uses ground truth keypoints from the DeepFashion2 dataset but predicts the depth.
    *   **To run:**
        ```bash
        python pdtk.py --anno_dirs path/to/anno/dir1 path/to/anno/dir2 --image_dirs path/to/image/dir1 path/to/image/dir2 --depth_model_path path/to/depth/model.pth --csv_output path/to/your/results.csv
        ```

## Dependencies

*   PyTorch
*   TorchVision
*   Matplotlib
*   NumPy
*   Pillow
*   OpenCV
*   SciPy
*   tqdm

The scripts also have a dependency on the [Depth-Anything-V2](https://github.com/LiheYoung/Depth-Anything) repository. It is assumed that this repository is cloned in the parent directory of this project.
