# 👁️ Behavioral & Visual Cheating Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0078D4?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

A state-of-the-art, **Multimodal AI Proctoring System** designed to detect suspicious behaviors during online examinations using real-time computer vision and behavioral analysis.

---

## 🚀 Key Features

### 🔹 Real-Time Visual Extraction
Utilizes **MediaPipe Face Mesh** and **Hands** to track 468+ facial landmarks and hand gestures with high precision and low latency.
- **Iris Tracking:** Precise eye-center and iris-to-eye-corner ratio extraction for gaze estimation.
- **Head Pose Estimation:** 6-DOF head orientation (Pitch, Yaw, Roll) to detect looking away.
- **Hand Presence:** Detects hand-object interactions (e.g., using a phone or covering the face).

### 🔹 Advanced Behavioral Analysis
Goes beyond simple face detection by analyzing temporal patterns:
- **Gaze Velocity:** Detects rapid, suspicious eye movements.
- **Eye Aspect Ratio (EAR):** Monitors blinks and prolonged eye closure.
- **Suspicious Flags:** Automatically flags events such as "Looking Away," "No Face Detected," or "Multiple Faces Detected."

### 🔹 Intelligent Risk Scoring
Combines visual features into a unified **Cheating Risk Score** (0-100%).
- **Model Fusion:** Uses a primary **Random Forest** model for speed and an optimized **CNN-LSTM** model for sequence-based analysis.
- **Temporal Windowing:** Analyzes behavior over a sliding window (10-30 frames) to reduce false positives.

### 🔹 Modern Exam Interface
A sleek, responsive frontend built for an authentic exam experience, featuring:
- **Integrated Proctoring Dashboard:** Real-time feedback for the user/proctor.
- **Visual Warning System:** Dynamic alerts and risk-level indicators (Safe, Warning, Critical).

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Webcam Feed] --> B[Visual Extractor]
    B --> C[Facial Landmarks]
    B --> D[Hand Landmarks]
    
    C --> E[Gaze Tracker]
    C --> F[Head Pose Estimator]
    
    E & F & D --> G[Behavior Analyzer]
    G --> H[Behavioral Flags]
    
    B & G --> I[Feature Fusion]
    I --> J[Risk Scorer Model]
    
    J --> K[Frontend Dashboard]
    H --> K
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sudikshabalajii/behavioral-visual-cheating-detection.git
cd behavioral-visual-cheating-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Models
Ensure you have the trained model artifacts in the `saved_models/` directory. If starting fresh, run the training pipeline:
```bash
python training/train.py
```

---

## 🏃 Usage

### Launching the Backend API
The Flask server handles both the real-time inference and serves the frontend interface.
```bash
python api/app.py
```
- **Frontend:** `http://localhost:5000/`
- **Health Check:** `http://localhost:5000/health`
- **Flag Logs:** `http://localhost:5000/flags`

---

## 📁 Project Structure

| Directory | Description |
| :--- | :--- |
| `api/` | Flask backend endpoints and server logic. |
| `webcam/` | Core CV logic: MediaPipe extraction, gaze tracking, and behavior analysis. |
| `frontend/` | Exam UI (HTML, CSS, JS) and proctoring dashboard. |
| `training/` | Scripts for training CNN-LSTM and Random Forest models. |
| `models/` | Feature engineering and data fusion utilities. |
| `scoring/` | Risk scoring algorithms and classification logic. |
| `saved_models/` | Serialized model artifacts (`.pkl`, `.keras`). |
| `data/` | Dataset and preprocessing scripts. |
| `plots/` | Generated training curves and evaluation matrices. |

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | ~98.5% | 0.98 | 0.99 | 0.98 |
| **XGBoost** | ~98.2% | 0.98 | 0.98 | 0.98 |
| **CNN+LSTM** | ~96.8% | 0.95 | 0.97 | 0.96 |

> [!NOTE]
> Performance metrics may vary based on the dataset used. The system currently prioritizes **Random Forest** for real-time inference due to its low latency.

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Contributors
- **Sudiksha Balaji** - *Initial Work*

---
<p align="center">
  Built with ❤️ for Academic Integrity
</p>
