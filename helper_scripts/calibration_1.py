import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import argparse

chessboard_rows = 7
chessboard_cols = 4

frame_size = (1280, 720)

chessboard_square_size = 3.11

def calibrate(input_path, show_images=True):
    # read image
    image_path_list = glob.glob(os.path.join(input_path, '*.jpg'))

    termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    global_pts = np.zeros((chessboard_cols*chessboard_rows, 3), np.float32)
    global_pts[:, :2] = np.mgrid[0:chessboard_rows, 0:chessboard_cols].T.reshape(-1,2)
    global_pts = global_pts * chessboard_square_size
    global_pts_list = []
    image_pts_list = []

    for cur_image_path in image_path_list:
        imgBGR = cv.imread(cur_image_path)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        corners_found, corners = cv.findChessboardCorners(imgGray, (chessboard_rows, chessboard_cols), None)

        if corners_found:
            global_pts_list.append(global_pts)
            corners_refined = cv.cornerSubPix(imgGray, corners, (11,11), (-1,-1), termination_criteria)

            image_pts_list.append(corners_refined)

            if show_images:
                cv.drawChessboardCorners(imgBGR, (chessboard_rows, chessboard_cols), corners_refined, corners_found)
                cv.imshow('Chessboard', imgBGR)
                cv.waitKey(500)
    cv.destroyAllWindows()

    # Calibrate
    reprojection_error, camMatrix, distCoeff, _, _ = cv.calibrateCamera(global_pts_list, image_pts_list, imgGray.shape[::-1], None, None)
    print("Camera Matrix:\n", camMatrix)
    print("Distortion Coefficients:\n", distCoeff)
    print("Reprojection Error [px]: {:.4f}".format(reprojection_error))

    # save calibration parameters
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, f'out\\{out_name}.txt')

    # Write to the file
    with open(paramPath, 'w') as file:
        file.write(f"Reprojection Error [px]: {reprojection_error:.4f}\n")
        file.write("Camera Matrix:\n")
        np.savetxt(file, camMatrix, fmt='%f')
        file.write("Distortion Coefficients:\n")
        np.savetxt(file, distCoeff, fmt='%f')

    print(f"Calibration parameters saved to {paramPath}")
    
    return


if __name__ == '__main__':

    out_name = "calibration"

    parser = argparse.ArgumentParser(description="Calibrate camera from a bunch of images containing a chessboard")
    
    parser.add_argument('-i', '--input', type=str, required=True, help='Set path to folder containing images')
    parser.add_argument('-o', '--output', type=str, required=False, help='Set output name')

    args = parser.parse_args()

    input_path = args.input

    if args.output is not None:
        out_name = args.output

    calibrate(input_path)