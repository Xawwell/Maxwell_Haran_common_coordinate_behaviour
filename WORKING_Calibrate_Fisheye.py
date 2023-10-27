import cv2
import numpy as np
import glob
import os

# SET PARAMETERS
calibration_images_loc = "D:\\Branco Lab\\big_checkerboard\\Big_checkerboard_images\\"
image_extension = '.png'
camera = 'Maxwell'
CHECKERBOARD = (13, 9)

# Specify the directory where you want to save the .npy file
calibration_directory = "D:\\Branco Lab\\calibration_data\\"

# Specify the filename for saving the .npy file
calibration_file = "Maxwell_calibration_map.npy"

# Create lists to store object points and image points
obj_points = []  # 3D points in real-world space
img_points = []  # 2D points in the image plane

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Find calibration images
image_paths = glob.glob(calibration_images_loc + f'*{image_extension}')

# Initialize a window
cv2.namedWindow('Corner Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Corner Detection', 800, 600)

# Iterate through calibration images
for image_path in image_paths:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If corners are found, add object points and image points
    if ret:
        obj_points.append(objp)
        img_points.append(corners)

        print(f"Found corners in {image_path}")

        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, CHECKERBOARD, corners, ret)

        img_height, img_width, _ = img_with_corners.shape
        aspect_ratio = img_width / img_height

        display_width = 800
        display_height = int(display_width / aspect_ratio)
        cv2.imshow('Corner Detection', cv2.resize(img_with_corners, (display_width, display_height)))

        cv2.waitKey(500)

cv2.destroyWindow('Corner Detection')

print("Corner detection completed.")

print("Calculating camera matrix K and distortion coefficients D...")
ret, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

if not os.path.exists(calibration_directory):
    os.makedirs(calibration_directory)

calibration_data = {
    'K': K,
    'D': D,
}
calibration_file_path = os.path.join(calibration_directory, calibration_file)
np.save(calibration_file_path, calibration_data)

print(f"Calibration completed. Calibration data saved to '{calibration_file_path}'")

loaded_calibration_data = np.load(calibration_file_path, allow_pickle=True).item()
K = loaded_calibration_data['K']
D = loaded_calibration_data['D']

output_directory = "D:\\Branco Lab\\calibration_output\\"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

...
for image_path in image_paths:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Get new camera matrix
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

    if new_K is not None and isinstance(new_K, np.ndarray):
        corrected_img = cv2.undistort(img, K, D, None, new_K)
    else:
        print("Error: Invalid new camera matrix.")
        continue

    # Resize the corrected image to match the original image dimensions
    corrected_img = cv2.resize(corrected_img, (w, h))

    # Print the dimensions to check if they match
    print(f"Original Image Dimensions: {img.shape}")
    print(f"Corrected Image Dimensions: {corrected_img.shape}")

    side_by_side = np.hstack((img, corrected_img))

    img_height, img_width, _ = side_by_side.shape
    aspect_ratio = img_width / img_height

    display_width = 1600
    display_height = int(display_width / aspect_ratio)
    cv2.imshow('Original vs. Corrected', cv2.resize(side_by_side, (display_width, display_height)))

    cv2.waitKey(1000)

    print(f"Saved corrected image: {output_directory}/{os.path.basename(image_path)}")
    cv2.imwrite(os.path.join(output_directory, os.path.basename(image_path)), corrected_img)

cv2.destroyWindow('Original vs. Corrected')

print("Calibration and correction of images completed.")
