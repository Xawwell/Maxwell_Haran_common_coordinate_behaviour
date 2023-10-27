import cv2
import os

# Specify the path to the video files and the output directory
video_paths = ['/Volumes/Extreme SSD/Branco Lab/big_checkerboard/calibration_checkerboard_rotate_every_10s_2023_10_24T13_37_02/cam_1.avi',
    '/Volumes/Extreme SSD/Branco Lab/big_checkerboard/calibration_checkerboard_rotate_every_10s_second_position_2023_10_24T13_41_45/cam_2.avi',
    '/Volumes/Extreme SSD/Branco Lab/big_checkerboard/calibration_checkerboard_rotate_every_10s_third_position_2023_10_24T13_49_59/cam_3.avi'
]
output_directory = '/Users/chenx/Desktop/Branco Lab/Shelter Choice Project/Maxwell-common-coordinate-behaviour/Maxwell_Haran_checkboard_maps'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for video_path in video_paths:
    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Couldn't open the video file: {video_path}")
        continue

    # Get the frames per second (fps) of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate the frame interval for every 10 seconds
    frame_interval = 10 * fps

    frame_count = 0
    while True:
        ret, frame = cap.read()

        # Break the loop if the video is finished
        if not ret:
            break

        # If the current frame is an interval of frame_interval, save it
        if frame_count % frame_interval == 0:
            # Extract the video file name without extension
            video_file_name = os.path.splitext(os.path.basename(video_path))[0]
            output_file_path = os.path.join(output_directory, f"{video_file_name}_frame_{frame_count}.png")
            cv2.imwrite(output_file_path, frame)
            print(f"Saved: {output_file_path}")

        frame_count += 1

    # Release the video capture object for this video
    cap.release()

print("Finished extracting frames from all videos.")
