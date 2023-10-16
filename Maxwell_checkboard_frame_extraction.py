import cv2
import os

# Specify the path to the video file and the output directory
video_path = '/Volumes/Extreme SSD/Branco Lab/video data/Checkboard/test_test_4_2023_06_01T14_50_57/cam.avi'
output_directory = '/Users/chenx/Desktop/Branco Lab/Shelter Choice Project/Maxwell-common-coordinate-behaviour/Maxwell_Haran_checkboard_maps'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Open the video using OpenCV
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Couldn't open the video file: {video_path}")
    exit()

# Get the frames per second (fps) of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Calculate the frame interval for every 10 seconds
frame_interval = 10 * fps

frame_count = 0
while True:
    ret, frame = cap.read()
    
    # Break the loop if video is finished
    if not ret:
        break

    # If the current frame is an interval of frame_interval, save it
    if frame_count % frame_interval == 0:
        output_file_path = os.path.join(output_directory, f"frame_{frame_count}.png")
        cv2.imwrite(output_file_path, frame)
        print(f"Saved: {output_file_path}")

    frame_count += 1

# Release the video capture object
cap.release()
print("Finished extracting frames.")