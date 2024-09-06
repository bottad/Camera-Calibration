import cv2

def calibrate_rectification(mtx_l, dist_l, mtx_r, dist_r, img_size, R, T, name="new"):

    R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, img_size, R, T, alpha=0)

    stereo_map_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R_l, P_l, img_size, cv2.CV_16SC2)
    stereo_map_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R_r, P_r, img_size, cv2.CV_16SC2)

    cv_file = cv2.FileStorage(f"data/out/stereo_map_{name}.xml", cv2.FILE_STORAGE_WRITE)


    cv_file.write('P_l', P_l)
    cv_file.write('P_r', P_r)
    cv_file.write('Q', Q)
    cv_file.write('stereo_map_l_x', stereo_map_l[0])
    cv_file.write('stereo_map_l_y', stereo_map_l[1])
    cv_file.write('stereo_map_r_x', stereo_map_r[0])
    cv_file.write('stereo_map_r_y', stereo_map_r[1])
    cv_file.write('roi_l', roi_l)
    cv_file.write('roi_r', roi_r)

    cv_file.release()