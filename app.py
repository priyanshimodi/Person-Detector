# app.py
import argparse
from detect import detect_from_video

def main():
    parser = argparse.ArgumentParser(description="Detect persons in a video using YOLOv8")
    parser.add_argument('--video', required=True, help="Path to input video file")
    parser.add_argument('--output', default='outputs/output.mp4', help="Path to save output video")
    args = parser.parse_args()

    detect_from_video(args.video, args.output)

if __name__ == '__main__':
    main()
