import cv2
import os
import argparse

pattern_size = (4,7) # Chessboard size

def filter_detectable(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Try to find the chessboard corners
    ret, _ = cv2.findChessboardCorners(gray, pattern_size, flags=cv2.CALIB_CB_FAST_CHECK)
    
    return ret  # Returns True if a chessboard is detected, False otherwise

def split_video_frames(video_path, start_frame, apply_filter=False):
    # Create directories if they do not exist
    directory_path = os.path.dirname(video_path)
    left_folder = os.path.join(directory_path, "left_images")
    right_folder = os.path.join(directory_path, "right_images")
    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)

    # Capture the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR]\tCannot open video file: {video_path}")
        return

    frame_count = 0
    saved_count = 0
    while True:

        ret, frame = cap.read()

        if not ret:
            break  # End of video
        
        if frame_count >= start_frame: 

            if apply_filter and not filter_detectable(frame):
                print(f"[INFO]\tFrame {frame_count} not detectable!")
                frame_count += 1
                continue

            # Get the dimensions of the frame
            _, width, _ = frame.shape

            # Calculate the middle index to split the frame
            mid_index = width // 2

            # Split the frame into left and right images
            left_image = frame[:, :mid_index]
            right_image = frame[:, mid_index:]

            # Save the left and right images
            left_image_path = os.path.join(left_folder, f"left_image_{frame_count}.jpg")
            right_image_path = os.path.join(right_folder, f"right_image_{frame_count}.jpg")
            
            cv2.imwrite(left_image_path, left_image)
            cv2.imwrite(right_image_path, right_image)

            print(f"[INFO]\tSaved frame {frame_count:04d} left and right images")
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"[INFO]\tVideo processing completed, saved {saved_count} images.")

def main():

    parser = argparse.ArgumentParser(description="Split stereo video into left and right images.")
    
    parser.add_argument('-i', '--input', type=str, required=True, help='Set input video path')
    parser.add_argument('-s', '--startframe', type=int, default=0, help='Set a start frame (ignore all frames before that frame)')
    parser.add_argument('-f', '--filter', action='store_true', help='Apply filter to detect frames')

    args = parser.parse_args()

    video_path = args.input
    start_frame = args.startframe
    apply_filter = args.filter

    print(f"Input video: {video_path}")
    print(f"Starting from frame: {start_frame}")
    if apply_filter:
        print("Only saving detectable images.")

    split_video_frames(video_path, start_frame, apply_filter)

if __name__ == '__main__':
    main()