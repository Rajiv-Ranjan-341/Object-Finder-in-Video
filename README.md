# 🔍 AI-Based Object Finder in Video using YOLOv8 and Streamlit

This project is a Streamlit-based web application that allows users to upload a video file (MP4), detect specific objects (e.g., license plates) using the YOLOv8 model, and optionally extract text using Tesseract OCR. It’s designed for efficient object detection and visualization frame-by-frame.

## 📁 Project Structure

prject B/
├── app.py # Main Streamlit application
├── requirements.txt # List of dependencies
└── yolov8n.pt # YOLOv8 model weights (can be custom-trained)

markdown
Copy
Edit

## 🎯 Features

- Upload `.mp4` video files via web UI.
- Detect specific objects (e.g., cars, license plates, persons).
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

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/project-b.git
cd project-b/prject\ B
2. Create Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
4. Set Up Tesseract OCR (Optional)
If you're using OCR features, install Tesseract OCR and set the path:

python
Copy
Edit
# In app.py
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
5. Run the App
bash
Copy
Edit
streamlit run app.py
🧪 How It Works
Extracts frames from the video every 0.5 seconds.

Runs YOLOv8 detection on each frame.

Matches object classes (e.g., "license plate") with user input.

Displays detection bounding boxes and timestamps in the Streamlit interface.

📦 Requirements (from requirements.txt)
streamlit

opencv-python

numpy

pytesseract

ultralytics

You can install them manually via:

bash
Copy
Edit
pip install streamlit opencv-python numpy pytesseract ultralytics
🧠 Model Information
This app uses yolov8n.pt, the smallest YOLOv8 model by default. You can replace it with a custom-trained model for license plates or other classes by training via:

bash
Copy
Edit
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
Then place the new weights file as yolov8n.pt in the project folder.


📄 License
This project is for educational and experimental purposes. Be sure to follow licensing of:

YOLOv8 by Ultralytics

Tesseract OCR

