import cv2
import time
import argparse
import yaml
import os

from src.calibrate_camera import calibrate_camera, save_camera_calibration
from src.stereo_calibration import stereo_calibrate
from src.calibrate_rectification import calibrate_rectification

# Load configuration from config.yaml
def load_config(config_path='data/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load config and set the pattern size and square size globally
config = load_config()
PATTERN_SIZE = (config['pattern_size']['width'], config['pattern_size']['height'])
CHESSBOARD_SQUARE_SIZE = config['chessboard_square_size']

def check_frame(frame):
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    success, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

    if success:
        cv2.drawChessboardCorners(frame_copy, PATTERN_SIZE, corners, success)
        cv2.imshow('Chessboard Detection', frame_copy)
        print("[INFO]\tPress Enter to confirm the frame or 's' to skip.")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('\r'):  # Enter key
                print("[INFO]\tFrame accepted.")
                return True
            elif key == ord('s'):
                print("[INFO]\tFrame skipped.")
                return False
    else:
        print("[WARNING]\tChessboard corners not detected.")
        return False

def check_stereo_frame(frame):
    _, width = frame.shape[:2]
    left_image = frame[:, :width // 2]
    right_image = frame[:, width // 2:]

    print("[INFO]\tChecking left image...")
    left_success = check_frame(left_image)
    if not left_success:
        return False

    print("[INFO]\tChecking right image...")
    right_success = check_frame(right_image)
    if not right_success:
        return False

    # If both left and right images are accepted
    print("[INFO]\tStereo frame accepted.")
    return True

def extract_calib_frames(video_path, stereo, folder_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR]\tCannot open video file {video_path}!")
        return

    selected_frames = []
    frame_count = 0

    while cap.isOpened():
        successful, frame = cap.read()
        if not successful:
            break

        cv2.imshow('Video', frame)
        key = cv2.waitKey(1) & 0xFF

        time.sleep(0.02)

        if key == ord(' '):
            if stereo:
                is_usable = check_stereo_frame(frame)
            else:
                is_usable = check_frame(frame)
            if is_usable:
                selected_frames.append(frame)
                print(f"[INFO]\tFrame {frame_count} selected for processing.")

        if key == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"[INFO]\tSelected frames for processing: {len(selected_frames)}")

    if stereo:
        for i, frame in enumerate(selected_frames):
            _, width = frame.shape[:2]
            left_image = frame[:, :width // 2]  # Left half of the frame
            right_image = frame[:, width // 2:]  # Right half of the frame
            left_image_path = f"{folder_path}/left_{i}.png"
            right_image_path = f"{folder_path}/right_{i}.png"
            cv2.imwrite(left_image_path, left_image)
            cv2.imwrite(right_image_path, right_image)
    else:
        for i, frame in enumerate(selected_frames):
            image_path = f"{folder_path}/frame_{i}.png"
            cv2.imwrite(image_path, frame)

    return

def load_images(folder_path, name):
    images = []

    counter = 0
    while True:
        file_name = f"{name}_{counter}.png"
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the file exists
        if not os.path.isfile(file_path):
            break
        
        # Load the image and append to the list
        image = cv2.imread(file_path)
        if image is not None:
            images.append(image)
        else:
            print(f"[WARNING]\tFailed to load image: {file_path}")

        counter += 1

    return images

if __name__ == "__main__":

    folder_path = "data/calib_images"
    is_stereo = True
    name = "new"

    parser = argparse.ArgumentParser(description="Camera Calibration Program")
    parser.add_argument('-v', '--video', type=str, help="Path to the video file for calibration")
    parser.add_argument('-s', '--stereo', action='store_true', help="Enable stereo calibration mode")
    parser.add_argument('-f', '--folder', type=str, help=f"Specify path to the calibration images folder (default: {folder_path}/<name>), setting this overwrites the /<name> ending!")
    parser.add_argument('-n', '--name', type=str, help=f"Specify a name of this calibration (default: {name})")

    args = parser.parse_args()

    if args.stereo:
        is_stereo = True

    if args.name is not None:
        name = args.name
        folder_path = f"{folder_path}/{name}"

    if args.folder is not None:
        folder_path = args.folder

    if args.video is not None:
        os.makedirs(folder_path, exist_ok=True)
        extract_calib_frames(args.video, is_stereo, folder_path=folder_path)

    # Perform intrinsic calibration(s)
    if is_stereo:
        left_images = load_images(folder_path, "left")

        rmse_l, mtx_l, dist_l = calibrate_camera(left_images, PATTERN_SIZE, CHESSBOARD_SQUARE_SIZE, False)
        save_camera_calibration(rmse_l, mtx_l, dist_l, f"data/out/calibration_{name}_left.txt")

        right_images = load_images(folder_path, "right")

        rmse_r, mtx_r, dist_r = calibrate_camera(right_images, PATTERN_SIZE, CHESSBOARD_SQUARE_SIZE, False)
        save_camera_calibration(rmse_r, mtx_r, dist_r, f"data/out/calibration_{name}_right.txt")



    else:
        images = load_images(folder_path, "frame")

        rmse, mtx, dist = calibrate_camera(images, PATTERN_SIZE, CHESSBOARD_SQUARE_SIZE)
        save_camera_calibration(rmse, mtx, dist, f"data/out/calibration_{name}.txt")

    
    # if stereo, perform extrinsic calibration
    if is_stereo:
        R, T = stereo_calibrate(mtx_l, dist_l, mtx_r, dist_r, left_images, right_images, PATTERN_SIZE, CHESSBOARD_SQUARE_SIZE, True)
        
        w = left_images[0].shape[1]
        h = left_images[0].shape[0]

        calibrate_rectification(mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T, name)