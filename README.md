# Camera Calibration

## Overview

The **Camera Calibration** project provides tools for camera calibration, including intrinsic calibration for a single camera, stereo calibration, and rectification. This project includes the following programs:

1. **calibration.py**: Performs intrinsic calibration for a single camera or stereo system.
2. **rectification.py**: Applies stereo rectification maps to correct for distortion and align the left and right image pairs.
3. **disparity_to_depth.py**: Computes depth maps from disparity maps using either projection matrices or a Q matrix.

## Installation

To install the required dependencies for the project, use the following command:

```bash
pip install -r requirements.txt
```

## Usage

For all scripts, run them using the -h flag to display all flag options.

### 1. calibration.py

This program performs intrinsic calibration for a single camera or a stereo system using eigther saved frames or a stereo video (left and right frames stitched together). It can be used for both single and stereo camera setups.

**Usage:**

```bash
python calibration.py -s -f <folder-path> -n <calibration-name> -v <path-to-video>
```

**Arguments:**

- `-s`: Enable stereo calibration mode (i.e., process stereo images for calibration).
- `-f <folder-path>`: Specify the path to the folder where calibration images are stored. Defaults to `data/calib_images/<name>`.
- `-n <calibration-name>`: Specify a name for this calibration. Defaults to `new`.
- `-v <path-to-video>`: Path to the video file for calibration.

**Example:**

```bash
python calibration.py -s -f data/calib_images/left_right -n my_calibration
```

### 2. rectification.py

This program applies stereo rectification maps to correct for distortion and align the left and right image pairs.

**Usage:**

```bash
python rectification.py -n <calibration-name> -l <left-images-folder> -r <right-images-folder> -o <output-folder>
```

**Arguments:**

- `-n <calibration-name>`: Name of the calibration file (e.g., `stereo_map_<name>.xml`). This file contains the rectification maps.
- `-l <left-images-folder>`: Folder containing the left images to rectify.
- `-r <right-images-folder>`: Folder containing the right images to rectify.
- `-o <output-folder>`: Folder to save the rectified images.

**Example:**

```bash
python rectification.py -n my_calibration -l data/left_images -r data/right_images -o data/output
```

### 3. disparity_to_depth.py

This program computes depth maps from disparity maps using either projection matrices or a Q matrix. It can generate both depth maps and heatmap visualizations.

**Usage:**

```bash
python disparity_to_depth.py -n <calibration-name> -i <input-folder> -o <output-folder> [-q]
```

**Arguments:**

- `-n <calibration-name>`: Name of the calibration file (e.g., `stereo_map_<name>.xml`). This file contains the projection matrices or Q matrix.
- `-i <input-folder>`: Folder containing the disparity `.npy` files.
- `-o <output-folder>`: Folder to save the depth maps and heatmaps.
- `-q`: Use Q matrix method for depth computation (default is to use projection matrices).

**Example:**

```bash
python disparity_to_depth.py -n my_calibration -i data/disparity -o data/depth -q
```