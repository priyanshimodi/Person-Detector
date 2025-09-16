import cv2
import os

video_path = "demo_videos/test.mp4"
output_folder = "outputs"

# Check if input video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit()

# Check if output folder exists; if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output folder: {output_folder}")

# Try opening the video
cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    print("Video opened successfully!")
else:
    print("Failed to open video!")

cap.release()
