import numpy as np
import os
import glob
import cv2


def find_checkerboard_corners(image_path, checkerboard, dark_threshold):
    """
    Detect checkerboard corners in the provided image.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[gray < dark_threshold] = 0

    ret, corners = cv2.findChessboardCorners(
        gray, checkerboard,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    )

    if ret:
        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), subpix_criteria)
        return True, corners_refined
    else:
        return False, None


def calibrate_fisheye(objpoints, imgpoints, img_shape):
    """
    Calibrate the camera using the provided object points and image points.
    """
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    try:
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints, imgpoints, img_shape, K, D, None, None,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        return True, K, D, rms
    except Exception as e:
        return False, None, None, str(e)


# Set parameters
calibration_images_loc = '/Users/chenx/Desktop/Branco Lab/Shelter Choice Project/Maxwell-common-coordinate-behaviour/Maxwell_Haran_checkboard_maps/'
CHECKERBOARD = (10,7)
dark_threshold = 20
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# Initialize lists for object points and image points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# Load images and detect checkerboard corners
images = glob.glob(calibration_images_loc + '*' + '.png')
for fname in images:
    success, corners = find_checkerboard_corners(fname, CHECKERBOARD, dark_threshold)
    if success:
        print(f'{fname}: successfully identified corners')
        objpoints.append(objp)
        imgpoints.append(corners)
    else:
        print(f'{fname}: failed to identify corners')

# Perform fisheye calibration
success, K, D, message = calibrate_fisheye(objpoints, imgpoints, img.shape[:2][::-1])
if success:
    print(f'Calibration successful. K={K} D={D}')
else:
    print(f'Calibration failed with error: {message}')

# TODO: Add undistortion testing and saving maps

# This is a starting point for the revised script. Further additions can be made based on feedback.
