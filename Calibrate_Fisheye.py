import numpy as np
import os
import glob
import cv2

'''
FISHEYE CALIBRATION AND CORRECTION CODE
Reference: https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
'''

# -------------------------- SET PARAMETERS --------------------------
calibration_images_loc = '/Users/chenx/Desktop/Branco Lab/Shelter Choice Project/Maxwell-common-coordinate-behaviour/Maxwell_Haran_checkboard_maps/'
image_extension = '.png'
camera = 'Haran'
CHECKERBOARD = (10,7)
dark_threshold = 20

print("Parameters set.")

# -------------------------- FIND CHECKERBOARD CORNERS --------------------------
CHECKERFLIP = tuple(np.flip(CHECKERBOARD, 0))
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None
objpoints = [] 
imgpoints = [] 
images = glob.glob(calibration_images_loc + '*' + image_extension)
print(f'Found {len(images)} images for calibration.')

for fname in images:
    img = cv2.imread(fname)
    
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    
    calib_image = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    calib_image[calib_image<dark_threshold] = 0

    ret, corners = cv2.findChessboardCorners(calib_image, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK)

    if ret == True:
        print(f'{fname}: successfully identified corners')
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(calib_image, corners, (11,11), (-1,-1), subpix_criteria)
        imgpoints.append(corners)
    else:
        print(f'{fname}: failed to identify corners')

print("Checkerboard corners identified.")

# -------------------------- GET CALIBRATION MATRICES K AND D --------------------------
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rms, _, _, _, _ = cv2.fisheye.calibrate(objpoints, imgpoints, calib_image.shape[::-1], K, D, None, None, calibration_flags, (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
print(f'Found {N_OK} valid images for calibration. K={K} D={D}')

# -------------------------- TEST CALIBRATION AND SAVE REMAPPINGS --------------------------
DIM = _img_shape[::-1]

for img_path in images:
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow("correction -- before and after", np.hstack((img, undistorted_img)))
    
    if cv2.waitKey(1000) & 0xFF == ord('q'):
       break

print("Calibration tested.")

# -------------------------- SAVE MAPS --------------------------
maps = np.zeros((calib_image.shape[0], calib_image.shape[1], 3)).astype(np.int16)
maps[:,:,0:2] = map1
maps[:,:,2] = map2
np.save(calibration_images_loc + 'Maxwell_fisheye_maps_' + camera + '.npy', maps)
print("Maps saved.")
