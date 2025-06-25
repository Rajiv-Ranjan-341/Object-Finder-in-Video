# 🔍 AI-Based Object Finder in Video using YOLOv8 and Streamlit

This project is a Streamlit-based web application that allows users to upload a video file (MP4), detect specific objects (e.g.,  cars, persons) using the YOLOv8 model, and optionally extract text using Tesseract OCR. It’s designed for efficient object detection and visualization frame-by-frame.


## 📁 Project Structure
```
Object-Finder-in-Video/
├── app.py # Main Streamlit application
├── requirements.txt # List of dependencies
└── yolov8n.pt # YOLOv8 model weights (can be custom-trained)
```

## 🎯 Features

- Upload `.mp4` video files via web UI.
- Detect specific objects (e.g., cars, persons).
- Use YOLOv8 model (via `ultralytics`) for real-time frame analysis.
- Extract frames every 0.5 seconds for efficient processing.
- Draw bounding boxes around detected objects with confidence scores.
- Use Tesseract OCR to extract text from objects (e.g., license plates).

## 🖥️ Tech Stack

- **Python 3**
- **Streamlit** - Frontend UI
- **OpenCV** - Video frame processing
- **Ultralytics YOLOv8** - Object detection model
- **Tesseract OCR** - Text extraction from detected regions
- **NumPy** - Array operations

## ⚙️ Setup Instructions

1. Clone the Repository

```bash
git clone https://github.com/Rajiv-Ranjan-341/Object-Finder-in-Video.git
cd Object-Finder-in-Video
```

2. Create Virtual Environment (Recommended)
   
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install Requirements
   
```bash
pip install -r requirements.txt
```

4. Set Up Tesseract OCR (Optional)
   
 If you're using OCR features, install Tesseract OCR and set the path:

## Install Tesseract
- Download the Tesseract installer from the official repo:
https://github.com/tesseract-ocr/tesseract/wiki 
- Direct link to the latest Windows installer:
https://github.com/UB-Mannheim/tesseract/wiki 
- Download and run the .exe installer (choose the default path or remember the install location).

##Add Tesseract to Your System PATH
- If you chose the default path, it will be something like:
C:\Program Files\Tesseract-OCR 
- Add this folder to your Windows PATH:
1. Press Win + S, type environment variables, and open "Edit the system environment variables".
2. Click "Environment Variables"
3. Under "System variables", find and select Path, then click "Edit".
4. Click "New" and add:
    C:\Program Files\Tesseract-OCR
5. Click OK on all dialogs.

# In app.py
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

5. Run the App
   
```bash
streamlit run app.py
```

🧪 How It Works
Extracts frames from the video every 0.5 seconds.

Runs YOLOv8 detection on each frame.

Matches object classes (e.g., " cars, persons") with user input.

Displays detection bounding boxes and timestamps in the Streamlit interface.

📦 Requirements (from requirements.txt)
-streamlit

-opencv-python

-numpy

-pytesseract

-ultralytics

You can install them manually via:

```bash
pip install streamlit opencv-python numpy pytesseract ultralytics
```
🧠 Model Information
This app uses yolov8n.pt, the smallest YOLOv8 model by default. You can replace it with a custom-trained model for other classes by training via:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```
Then place the new weights file as yolov8n.pt in the project folder.


📄 License
This project is for educational and experimental purposes. Be sure to follow licensing of:
-YOLOv8 by Ultralytics
-Tesseract OCR

