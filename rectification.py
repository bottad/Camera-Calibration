import cv2
import numpy as np
import os
import argparse

def main(name, input_folder_l, input_folder_r, output_folder):
    print("[INFO]\tLoad rectification map.")
    # Open the stereo map XML file
    cv_file = cv2.FileStorage()
    cv_file.open(f"data/out/stereo_map_{name}.xml", cv2.FILE_STORAGE_READ)

    # Read stereo maps
    stereo_map_l_x = cv_file.getNode('stereo_map_l_x').mat()
    stereo_map_l_y = cv_file.getNode('stereo_map_l_y').mat()
    stereo_map_r_x = cv_file.getNode('stereo_map_r_x').mat()
    stereo_map_r_y = cv_file.getNode('stereo_map_r_y').mat()

    # Close the XML file
    cv_file.release()

    # Ensure output folders exists
    if not os.path.exists(f"{output_folder}/rectified_left"):
        os.makedirs(f"{output_folder}/rectified_left")
    if not os.path.exists(f"{output_folder}/rectified_right"):
        os.makedirs(f"{output_folder}/rectified_right")

    counter = 0

    print("[INFO]\tRectifying images...\n")

    while True:
        if counter % 100 == 0:
            print(f"      \t... Processed {counter} images...", end="\r")
        # Load both images
        image_path_r = f"{input_folder_r}/right_image_{counter}.jpg"
        image_path_l = f"{input_folder_l}/left_image_{counter}.jpg"
        img_r = cv2.imread(image_path_r)
        img_l = cv2.imread(image_path_l)

        # quit process when all images are processed
        if img_r is None or img_l is None:
            print(f"[INFO]\tAll {counter} images rectified.")
            break

        # Apply stereo rectification maps
        rect_img_l = cv2.remap(img_l, stereo_map_l_x, stereo_map_l_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        rect_img_r = cv2.remap(img_r, stereo_map_r_x, stereo_map_r_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        # Save the rectified images to the output folder
        output_path_l = f"{output_folder}/rectified_left/rectified_left_image_{counter}.jpg"
        output_path_r = f"{output_folder}/rectified_right/rectified_right_image_{counter}.jpg"

        cv2.imwrite(output_path_l, rect_img_l)
        cv2.imwrite(output_path_r, rect_img_r)

        # Validate saved images by reloading them
        saved_img_l = cv2.imread(output_path_l)
        saved_img_r = cv2.imread(output_path_r)

        if saved_img_l is None:
            print(f"[ERROR]\tSaved left image is faulty: {output_path_l}")
            continue
        if saved_img_r is None:
            print(f"[ERROR]\tSaved right image is faulty: {output_path_r}")
            continue

        # Increment the counter
        counter += 1
    print("                                                                                    ", end="\r")
    print(f"[INFO]\t... Rectification of {counter} images complete!")
    return

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Stereo image rectification.")
    parser.add_argument('-n', '--name', default="new", help="Name identifier for the stereo rectification maps (default: new).")
    parser.add_argument('-r', '--right', help="Input folder containing the right images.", required=True)
    parser.add_argument('-l', '--left', help="Input folder containing the left images.", required=True)
    parser.add_argument('-o', '--out', help="Output folder to save the rectified images.", required=True)
    args = parser.parse_args()

    main(args.name, args.left, args.right, args.out)

# Example Usage:
# python rectification.py -n run_2 -r "D:/fft out/start_dataset/Stereo_images_right" -l "D:\fft out\start_dataset\Stereo_images_left" -o "D:\fft out\start_dataset\rectification"