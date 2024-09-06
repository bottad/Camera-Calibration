import os
from PIL import Image
import numpy as np
import argparse

def check_images_in_folder(folder_path):
    faulty_images = []

    # Walk through the directory to find images
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                file_path = os.path.join(root, file)
                try:
                    # Try to open the image and convert it to a NumPy array
                    img = Image.open(file_path)
                    img = img.convert('RGB')  # Convert to RGB to ensure consistency
                    np.array(img).astype(np.uint8)  # Attempt conversion to NumPy array
                    print(f"{file}: OK")
                except Exception as e:
                    # If any exception occurs, log the file as faulty
                    print(f"{file}: Faulty ({str(e)})")
                    faulty_images.append(file_path)

    # Output the list of faulty images
    if faulty_images:
        print("\nFaulty Images:")
        for img in faulty_images:
            print(img)
    else:
        print("No faulty images found.")

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Check for faulty images in a folder.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input folder containing images.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Check images in the specified folder
    check_images_in_folder(args.input)
