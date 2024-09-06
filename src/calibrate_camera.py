import cv2
import numpy as np

def calibrate_camera(images, pattern_size, square_size, visualize=True):
    # Criteria used by checkerboard pattern detector
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Define arrays to store object points and image points
    object_points = []
    image_points = []

    # Prepare the object points grid
    obj_p = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    obj_p *= square_size

    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(gray_img, pattern_size, None)  # Directly use pattern_size

        if success:
            corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), termination_criteria)
            if visualize:
                cv2.drawChessboardCorners(img, pattern_size, corners2, success)
                cv2.imshow('Chessboard Detection', img)
                cv2.waitKey(500)
            object_points.append(obj_p)
            image_points.append(corners)

    cv2.destroyAllWindows()

    # Perform camera calibration
    if len(object_points) > 0 and len(image_points) > 0:
        rmse, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, gray_img.shape[::-1], None, None
        )
        #opt_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, gray_img.shape[::-1], 1, gray_img.shape[::-1])
        print("Camera Matrix:\n", camera_matrix)
        #print("Optimized Camera Matrix:\n", opt_camera_matrix)
        print("Distortion Coefficients:\n", distortion_coeffs)
        print("Reprojection Error [px]: {:.4f}".format(rmse))
        return rmse, camera_matrix, distortion_coeffs
    else:
        print("[ERROR]\tNo valid chessboard corners found.")
        return None, None, None
    

def save_camera_calibration(rmse, camera_matrix, distortion_coeffs, file_path):
    # Write to the file
    with open(file_path, 'w') as file:
        file.write(f"Reprojection Error [px]: {rmse:.4f}\n")
        file.write("Camera Matrix:\n")
        np.savetxt(file, camera_matrix, fmt='%f')
        file.write("Distortion Coefficients:\n")
        np.savetxt(file, distortion_coeffs, fmt='%f')

    print(f"Calibration parameters saved to {file_path}")