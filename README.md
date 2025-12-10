# 💪 RepCheck - AI-Powered Lifting Form Analysis

RepCheck uses computer vision to analyze your squat form and provide instant feedback on depth, back angle, and knee alignment. Built to help lifters prevent injuries and improve technique.

![RepCheck Demo](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.12-blue)

## 🎯 Problem Statement

Back injuries from improper lifting cost US companies $70 billion annually. Personal trainers are not always available. RepCheck provides real-time form analysis accessible to anyone with a camera.

## ✨ Features

- **📸 Photo Analysis** - Upload images or use camera (works on iPhone!)
- **🤖 AI-Powered Detection** - YOLOv8 pose estimation for accurate form checking
- **📊 Instant Feedback** - Get pass/fail results on:
  - Squat depth (hips below parallel)
  - Back angle (staying upright)
  - Knee alignment (tracking over feet)
- **🖼️ Visual Annotations** - See exactly where form needs improvement
- **📱 Mobile-Friendly** - Works on any device with a browser
- **💾 Downloadable Reports** - Save annotated images for progress tracking

## 🏗️ Architecture

```
RepCheck
├── Backend (FastAPI)
│   ├── YOLOv8 Pose Detection
│   ├── Form Analysis Logic
│   └── Image Annotation
│
├── Frontend (Streamlit)
│   ├── Photo Upload
│   ├── Camera Capture
│   └── Results Display
│
└── AI Model (YOLOv8n-pose)
    └── 17 keypoint detection
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- Webcam (optional, for live capture)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/emresendev/repcheck.git
cd repcheck
```

2. **Create virtual environment**

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Running RepCheck

**Terminal 1 - Start Backend:**

```bash
python backend/app.py
```

**Terminal 2 - Start Frontend:**

```bash
streamlit run frontend/app.py
```

**Open your browser to:** `http://localhost:8501`

## 📱 How to Use

### Setup

1. Position camera at hip height, 3-4 feet away
2. Ensure **side angle view** (90° to your body)
3. Make sure full body is visible

### Capture

1. Use Bluetooth remote or camera self-timer
2. Get into bottom position of squat
3. Capture photo

### Analyze

1. Upload photo or use camera
2. Click "Analyze Form"
3. Review instant feedback
4. Download annotated image

## 🧪 Technical Details

### Tech Stack

**Backend:**

- FastAPI - Modern Python web framework
- YOLOv8 - State-of-the-art pose estimation
- OpenCV - Image processing and annotation
- Pillow - Image manipulation

**Frontend:**

- Streamlit - Rapid web app development
- Mobile-responsive design
- Works on iOS and Android

**AI Model:**

- YOLOv8n-pose (nano version)
- 17 keypoint detection per person
- ~90ms inference time on CPU
- No GPU required

### Form Analysis Checks

**1. Depth Check**

- Measures hip position relative to knee
- **Pass:** Hips below parallel (hip_y > knee_y)
- **Fail:** Squat too shallow

**2. Back Angle Check**

- Calculates angle between shoulder-hip-knee
- **Pass:** 45° - 90° (upright posture)
- **Fail:** Leaning too far forward

**3. Knee Alignment Check**

- Measures knee position relative to ankle
- **Pass:** Knees tracking over feet
- **Warning:** Knees caving in or pushing too far forward

## 📊 API Documentation

### Endpoints

**Health Check**

```
GET /
Response: {"status": "healthy", "message": "RepCheck API is running!", "version": "1.0.0"}
```

**Analyze Form**

```
POST /api/analyze
Body: multipart/form-data with image file
Response: {
  "success": true,
  "all_pass": false,
  "feedback": [...],
  "annotated_image": "/images/processed/..."
}
```

**Get Stats**

```
GET /api/stats
Response: {
  "total_analyzed": 10,
  "total_processed": 10
}
```

## 🗂️ Project Structure

```
repcheck/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── form_analyzer.py    # YOLOv8 analysis logic
│   └── utils.py
│
├── frontend/
│   └── app.py              # Streamlit interface
│
├── data/
│   ├── uploads/            # User uploaded images
│   └── processed/          # Annotated images
│
├── models/
│   └── yolov8n-pose.pt     # Pre-trained model (auto-downloads)
│
├── requirements.txt
└── README.md
```

## 🎯 Future Enhancements

### Phase 2

- [ ] Video analysis (frame-by-frame)
- [ ] Multiple exercise types (deadlift, bench press)
- [ ] Rep counting
- [ ] Progress tracking over time

### Phase 3

- [ ] Real-time webcam analysis
- [ ] Mobile app (React Native)

### Phase 4

- [ ] User accounts and history
- [ ] Share results with trainers
- [ ] Community leaderboards
- [ ] Exercise recommendations based on form

## 🙏 Acknowledgments

- **Ultralytics YOLOv8** - Pose estimation model
- **FastAPI** - Backend framework
- **Streamlit** - Frontend framework

## 📧 Contact

Built by Monica Nieckula

- GitHub: @emresendev
