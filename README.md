# Person Detector (YOLOv8 Real-Time Object Detection) 

This project performs real-time object detection using the YOLOv8 model by Ultralytics.

It allows detection of people and other objects in videos or through webcam input, drawing bounding boxes and saving results. 

---

## 🚀 Features  
- **Detects multiple objects using YOLOv8**  
  - The main class is person which is identified by the model and bounding boxes are drawn around it with the confidence score.
  -  Other object classes are also identified and results showed. 
- **Supports video file input and webcam feed**  
- **Result Display**  
  - Outputs processed video with bounding boxes, class label and confidence score
  - Other classes are displayed on top right with count

---

## 📂 Project Structure  
Person-Detector
├── app.py                     # Main script to run detection
├── detect.py                  # Detection logic (runs YOLO,processes frames)
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview, usage instructions
├── demo_videos/               # Input videos
│   └── test.mp4
├── outputs/                   # Output video 
│   └──  result.mp4

---

## ⚡ How It Works  
1. Setup by installing the required Python libraries:
    - ultralytics → for YOLOv8
    - opencv-python → for video processing and bounding box drawing
2. Input Video
    - You place a video file inside the demo_videos folder.
    - This is the input video you want to run object detection on.
3. YOLOv8 Loads
    - When you run the app.py file, it loads the YOLOv8 model
4. Frame-by-Frame Detection
    - Your code reads the input video frame by frame using OpenCV.
    - For each frame: 
        - YOLO predicts bounding boxes, class labels, and confidence scores.
        - The code draws boxes and labels on detected objects.
        - It counts how many people are detected.
        - Displays the processed frame in a window (live view).
5. Output Saving
    - The final processed video is saved in outputs/result.mp4
