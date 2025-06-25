import streamlit as st
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import tempfile
import os

# Set Tesseract cmd path if needed (uncomment and set your path)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def extract_frames(video_path, interval=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    timestamps = []
    count = 0
    success, frame = cap.read()
    while success:
        if int(count % int(fps * interval)) == 0:
            frames.append(frame.copy())
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        success, frame = cap.read()
        count += 1
    cap.release()
    return frames, timestamps

def detect_objects(frame, model, target_class):
    results = model(frame)
    boxes = results[0].boxes
    names = results[0].names if hasattr(results[0], 'names') else model.names
    detected = []
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = names[cls_id] if names and cls_id < len(names) else str(cls_id)
        conf = float(box.conf[0])
        if cls_name.lower() == target_class.lower():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected.append({
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'class': cls_name
            })
    return detected

def main():
    st.title("Object Finder in Video (AI)")
    st.write("Upload an MP4 video and select the object class to search for.")

    uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4"])

    # List of COCO classes for YOLOv8n
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    target_class = st.selectbox("Select object class to find", coco_classes, index=0)

    if uploaded_file and target_class:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(uploaded_file.read())
            video_path = tmpfile.name

        st.info("Loading YOLOv8 model (this may take a few seconds)...")
        model = YOLO("yolov8n.pt")

        st.info("Extracting frames from video...")
        frames, timestamps = extract_frames(video_path, interval=0.5)
        st.success(f"Extracted {len(frames)} frames.")

        found = False
        results = []
        st.info(f"Processing frames for '{target_class}' detection...")
        progress = st.progress(0)
        for idx, (frame, ts) in enumerate(zip(frames, timestamps)):
            detections = detect_objects(frame, model, target_class)
            if detections:
                found = True
                frame_boxed = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    conf = det['conf']
                    cls_name = det['class']
                    cv2.rectangle(frame_boxed, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame_boxed, f"{cls_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                results.append((ts, frame_boxed, detections))
            progress.progress((idx+1)/len(frames))
        progress.empty()
        os.remove(video_path)

        if found:
            st.success(f"Found {len(results)} frames with '{target_class}'!")
            for ts, frame, detections in results:
                st.write(f"**Time:** {ts:.2f} seconds | **Detections:** {[f'{d['class']} ({d['conf']:.2f})' for d in detections]}")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        else:
            st.warning(f"No '{target_class}' found in the video.")

if __name__ == "__main__":
    main() 