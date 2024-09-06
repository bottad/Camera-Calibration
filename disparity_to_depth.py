import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import argparse
import os

def compute_dept_from_disparity_and_projection(P1, P2, disparity_npy_path, output_depth_path=None, heatmap_file_path=None, colormap='inferno_r', vmin=0, vmax=3):
    """
    Compute the depth map from the disparity .npy file and save it to the specified output paths.
    
    Args:
    - P1: 3x4 projection matrix for the left camera.
    - P2: 3x4 projection matrix for the right camera.
    - disparity_npy_path: Path to the disparity .npy file.
    - output_depth_path: Path where the output depth map should be saved as a floating-point image (default: None).
    - heatmap_file_path: Path where the heatmap image will be saved (default: None).
    - colormap: The colormap to use for heatmap visualization (default: 'plasma').
    - vmin: Minimum value for heatmap scaling (default: 0).
    - vmax: Maximum value for heatmap scaling (default: 3).
    """
    
    # Load the disparity map from the .npy file
    disparity_map = np.load(disparity_npy_path).astype(np.float32)
    
    if disparity_map is None:
        raise FileNotFoundError(f"Disparity .npy file not found at {disparity_npy_path}")

    f_x = P1[0, 0]  # Focal length in x-direction
    cx_left = P1[0, 2]  # Principal point x-coordinate of the left camera
    cx_right = P2[0, 2]  # Principal point x-coordinate of the right camera
    baseline = np.abs(P2[0, 3] / f_x) # Baseline between the two cameras

    # Calculate depth:
    adjusted_disparity = np.abs(disparity_map + (cx_right - cx_left))
    epsilon = 1e-6  # Small constant to avoid division by zero
    adjusted_disparity[adjusted_disparity == 0] = epsilon
    depth_map = (f_x * baseline) / adjusted_disparity

    # Save depth map as a floating-point image if output_depth_path is provided
    if output_depth_path is not None:
        depth_pil_image = Image.fromarray(depth_map.astype(np.float32), mode='F')
        depth_pil_image.save(output_depth_path)
        #print(f'Saved depth map: {output_depth_path}')

    # Save heatmap visualization of the depth map if heatmap_file_path is provided
    if heatmap_file_path is not None:
        plt.imshow(depth_map, cmap=colormap, vmin=vmin, vmax=vmax)
        
        # Remove axes
        plt.axis('off')

        # Save the heatmap without any additional space around the image
        plt.savefig(heatmap_file_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        #print(f'Saved heatmap: {heatmap_file_path}')


def compute_depth_from_disparity_and_Q(disparity_npy_path, Q_matrix, output_depth_path=None, heatmap_file_path=None, colormap='inferno_r', vmin=0, vmax=3):
    """
    Compute the depth map from the disparity .npy file using the Q matrix and save it to the specified output paths.
    
    Args:
    - disparity_npy_path: Path to the disparity .npy file.
    - Q_matrix: 4x4 Q matrix used for disparity to 3D conversion.
    - output_depth_path: Path where the output depth map should be saved as a floating-point image (default: None).
    - heatmap_file_path: Path where the heatmap image will be saved (default: None).
    - colormap: The colormap to use for heatmap visualization (default: 'plasma').
    - vmin: Minimum value for heatmap scaling (default: 0).
    - vmax: Maximum value for heatmap scaling (default: 3).
    """
    
    # Load the disparity map from the .npy file
    disparity_map = np.load(disparity_npy_path).astype(np.float32)
    
    if disparity_map is None:
        raise FileNotFoundError(f"Disparity .npy file not found at {disparity_npy_path}")

    # Create a depth map
    h, w = disparity_map.shape
    disparity_map = disparity_map.reshape(-1, 1)
    ones = np.ones((h * w, 1), dtype=np.float32)
    
    # Create a homogeneous disparity map (add the third dimension for disparity)
    disparity_map_homogeneous = np.hstack((disparity_map, ones))
    
    # Compute 3D points using the Q matrix
    points_3d_homogeneous = np.dot(Q_matrix, disparity_map_homogeneous.T).T
    
    # Convert back from homogeneous coordinates
    points_3d = points_3d_homogeneous[:, :3] / points_3d_homogeneous[:, 3][:, np.newaxis]
    
    # Extract depth (Z coordinate)
    depth_map = points_3d[:, 2].reshape(h, w)
    
    # Save depth map as a floating-point image if output_depth_path is provided
    if output_depth_path is not None:
        depth_pil_image = Image.fromarray(depth_map.astype(np.float32), mode='F')
        depth_pil_image.save(output_depth_path)
        #print(f'Saved depth map: {output_depth_path}')

    # Save heatmap visualization of the depth map if heatmap_file_path is provided
    if heatmap_file_path is not None:
        plt.imshow(depth_map, cmap=colormap, vmin=vmin, vmax=vmax)
        
        # Remove axes
        plt.axis('off')

        # Save the heatmap without any additional space around the image
        plt.savefig(heatmap_file_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        #print(f'Saved heatmap: {heatmap_file_path}')


def main(args):
    # Load calibration data
    cv_file_path = f"data/out/stereo_map_{args.name}.xml"
    if not os.path.exists(cv_file_path):
        raise FileNotFoundError(f"Calibration file not found: {cv_file_path}")
    
    cv_file = cv2.FileStorage(cv_file_path, cv2.FILE_STORAGE_READ)
    
    P1 = cv_file.getNode('P_l').mat()
    P2 = cv_file.getNode('P_r').mat()
    Q = cv_file.getNode('Q').mat()
    
    cv_file.release()
    
    input_folder = args.input
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    counter = 0

    print("[INFO]\tComputing depth images from disparity maps...")

    while True:

        if counter % 10 == 0:
            print(f"      \t... Processed {counter} maps...", end="\r")

        npy_path = f"{input_folder}/rectified_left_{counter}.npy"

        if not os.path.exists(npy_path):
            print(f"No more files found. Exiting loop.")
            break

        output_depth_path = f"{output_folder}/{counter}_depth.tif"
        heatmap_file_path = f"{output_folder}/{counter}_heatmap.png"

        if args.use_q:
            compute_depth_from_disparity_and_Q(Q, npy_path, output_depth_path, heatmap_file_path)
        else:
            compute_dept_from_disparity_and_projection(P1, P2, npy_path, output_depth_path, heatmap_file_path)

        counter += 1
    print("                                                                                    ", end="\r")
    print(f"[INFO]\t... Conversion of {counter} disparity maps complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute and save depth maps from disparity images.")
    
    parser.add_argument('-q', '--use_q', action='store_true', help='Use Q matrix method (default: projection matrices)')
    parser.add_argument('-n', '--name', required=True, help='Name of the calibration file (located in data/out/stereo_map_<name>.xml)')
    parser.add_argument('-i', '--input', required=True, help='Folder containing the disparity .npy files')
    parser.add_argument('-o', '--output', required=True, help='Folder to save the depth results to')
    
    args = parser.parse_args()
    
    main(args)