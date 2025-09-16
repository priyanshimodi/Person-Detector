import cv2
from ultralytics import YOLO

# COCO classes used by YOLOv8 (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def detect_from_video(video_path, output_path=None, display_width=1280, display_height=720):
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Unable to open video {video_path}")

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        results = model.predict(frame, save=False)

        # Count detected objects per class
        counts = {}

        for detection in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = detection
            cls = int(cls)
            label = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else f"class_{cls}"
            counts[label] = counts.get(label, 0) + 1

            # Draw bounding box for person only (class 0)
            if cls == 0:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Optionally, draw boxes for other classes too (comment this if too cluttered)
                # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                pass

        # Display counts on the frame (top-left corner, stacked)
        start_y = 30
        for i, (label, count) in enumerate(sorted(counts.items())):
            text = f"{label}: {count}"
            # Bigger font, bold (thickness 3)
            cv2.putText(frame, text, (10, start_y + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Resize frame proportionally for display
        height, width = frame.shape[:2]
        scale_w = display_width / width
        scale_h = display_height / height
        scale = min(scale_w, scale_h)

        resized_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

        cv2.imshow("YOLO Object Detection", resized_frame)

        if out:
            out.write(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("User pressed 'q'. Exiting...")
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
