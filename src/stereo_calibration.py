import cv2
import numpy as np

def stereo_calibrate(mtx_l, dist_l, mtx_r, dist_r, images_l, images_r, pattern_size, square_size, visualize=True):
 
    #change this if stereo calibration not good.
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    # Prepare the object points grid
    obj_p = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    obj_p *= square_size

    image_points_left = [] 
    image_points_right = []
 
    object_points = []
 
    for frame_l, frame_r in zip(images_l, images_r):
        gray_img_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_image_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        success_l, corners_l = cv2.findChessboardCorners(gray_img_l, pattern_size, None)
        success_r, corners_r = cv2.findChessboardCorners(gray_image_r, pattern_size, None)
 
        if success_l == True and success_r == True:
            corners_l = cv2.cornerSubPix(gray_img_l, corners_l, (11, 11), (-1, -1), termination_criteria)
            corners_r = cv2.cornerSubPix(gray_image_r, corners_r, (11, 11), (-1, -1), termination_criteria)

            if visualize:

                # Draw chessboard corners on both images
                cv2.drawChessboardCorners(frame_l, pattern_size, corners_l, success_l)
                cv2.drawChessboardCorners(frame_r, pattern_size, corners_r, success_r)

                combined_frame = cv2.hconcat([frame_l, frame_r])
                cv2.imshow('Combined Image', combined_frame)
                cv2.waitKey(500)
 
            object_points.append(obj_p)
            image_points_left.append(corners_l)
            image_points_right.append(corners_r)

    cv2.destroyAllWindows()
 
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    rmse, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(object_points, image_points_left, image_points_right, mtx_l, dist_l,
                                                                 mtx_r, dist_r, gray_img_l.shape[::-1], criteria = termination_criteria, flags = stereocalibration_flags)
 
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", T)
    print("Reprojection Error [px]: {:.4f}".format(rmse))
    return R, T