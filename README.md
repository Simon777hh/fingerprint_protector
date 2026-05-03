# Fingerprint Protector 🛡️

> Gesture-aware fingerprint privacy protection — automatically detects gestures and blurs exposed fingerprint regions in photos

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red.svg)](https://pytorch.org/)

## 📖 Overview

When sharing photos on social media, fingerprints can be captured by high-resolution cameras and potentially misused. This project uses **gesture recognition** to detect whether the current hand gesture exposes fingerprints (e.g., ✌️ peace sign), then automatically locates fingertip regions and applies **Gaussian blur** to protect fingerprint privacy.

## ✨ Features

- 🖐️ **Gesture Recognition** — Uses MediaPipe to extract 21 hand landmarks and a 1D-CNN to classify 18 gesture types
- 🎯 **Smart Mask Generation** — Dynamically sizes elliptical masks based on finger distance from camera
- 🔒 **Two Protection Modes** — Fast Gaussian blur (default) or diffusion-based inpainting (optional)
- 📸 **Camera Capture** — Built-in camera support for real-time photo protection
- ⚡ **Lightweight** — Blur mode runs on CPU with no GPU required

## 🗂️ Project Structure

```
├── fingerprint_utils/           # Core module
│   ├── __init__.py
│   ├── feature_extractor.py     # MediaPipe hand landmark extraction
│   ├── gesture_classifier.py    # 1D-CNN gesture classification (18 classes)
│   ├── mask_generator.py        # Elliptical mask generation for fingertips
│   ├── blur_protector.py        # Gaussian blur protector
│   └── image_protector.py       # Diffusion-based inpainting protector (optional)
├── train_gesture_classifier.py  # Training script for gesture model
├── prepare_data.py              # Data preprocessing (HaGRID dataset)
├── protect_image.py             # Main entry point
├── gesture_classifier.pth       # Trained model weights
├── gesture_names.json           # Gesture class labels
└── gesture_data.npz             # Preprocessed training data
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- A webcam (for camera capture mode)

### Installation

```
# Clone the repository
git clone <(https://github.com/Simon777hh/fingerprint_protector.git)>
cd <fingerprint_protector>

# Install dependencies
pip install opencv-python numpy mediapipe torch scikit-learn tqdm pillow

# (Optional) For diffusion inpainting mode
pip install diffusers
```
Usage
```
1. Protect an existing image:

python protect_image.py path/to/your/photo.jpg

2. Capture and protect from camera:

python protect_image.py
**When prompted, press 'y' to use camera, SPACE to capture, ESC to quit**

3. Use in your own code:

python
from fingerprint_utils import HandFeatureExtractor, GestureClassifier, MaskGenerator, BlurProtector
import cv2

# Initialize
extractor = HandFeatureExtractor()
classifier = GestureClassifier()
mask_gen = MaskGenerator()
protector = BlurProtector()

# Process image
image = cv2.imread("photo.jpg")
features, landmarks = extractor.extract_features_from_array(image)
fingers = classifier.get_exposed_fingers(features)
mask = mask_gen.create_mask(image, landmarks, fingers)
protected = protector.protect(image, mask, blur_strength=51)

cv2.imwrite("protected.jpg", protected)
```
## 🎮 Supported Gestures
Gesture	Exposed Fingers	Protected
| Gesture | Exposed Fingers | Protected |
|---------|:---------------:|:---------:|
| ✌️ Peace | Index, Middle | ✅ |
| 👍 Like | Thumb | ✅ |
| 👆 One | Index | ✅ |
| ✋ Palm | All five | ✅ |
| 🤘 Rock | Index, Pinky | ✅ |
| ✊ Fist | None | ❌ |
| 👌 OK | Middle, Ring, Pinky | ✅ |

See gesture_names.json for the full list of 18 supported gestures.

## 🔧 How It Works
Hand Detection — MediaPipe extracts 21 hand landmarks (63-dim features)

Gesture Classification — 1D-CNN predicts the gesture from landmark coordinates

Mask Generation — Elliptical masks are placed on exposed fingertips with distance-aware sizing

Protection — Gaussian blur is applied to masked regions with smooth edge blending

Input Image → Hand Landmarks → Gesture → Finger Mask → Blurred Output
## 📊 Model Training
The gesture classifier was trained on the HaGRID dataset:

### Step 1: Prepare data
python prepare_data.py

### Step 2: Train model
python train_gesture_classifier.py
Component	Detail
Architecture	1D-CNN (3 conv layers + 2 FC layers)
Input	63-dim hand landmarks
Classes	18 gestures
Accuracy	~95% on validation set
## 🛠️ Tech Stack
Hand Detection: MediaPipe

Deep Learning: PyTorch

Image Processing: OpenCV

Dataset: HaGRID (Hand Gesture Recognition Image Dataset)

## 🚧 TODO / Limitations
Currently a personal student project with the following known limitations:

| # | Limitation | Description |
|---|------------|-------------|
| 1 | Single hand only | Only detects and protects one hand per image. Multi-hand support is not yet implemented. |
| 2 | Camera resolution | Camera capture uses fixed 1280×720 resolution with no configuration options. |
| 3 | No batch processing | Processes one image at a time. No folder batch mode available. |
| 4 | Fixed blur strength | Gaussian blur kernel size is hardcoded (default 51). No UI or CLI parameter to adjust. |
| 5 | No GUI | Command-line only. A simple GUI would improve usability. |
| 6 | Diffusion mode offline | The optional diffusion inpainting requires manual model download and ~4GB GPU memory. |
| 7 | Limited gesture coverage | Only 18 gestures supported. Custom gestures require retraining. |
| 8 | No video support | Static images only. Real-time video stream protection is not available. |

Planned Features:
Multi-hand detection and protection

Real-time camera preview with overlay

Batch image processing

Simple GUI (e.g., Gradio / Tkinter)

Adjustable blur strength via CLI argument

Export mask as separate file option

## 📝 License
This project is licensed under the MIT License — see the LICENSE file for details.

## 🙋‍♂️ Author
Simon777hh

A student project built for learning and portfolio

## ⚠️ Disclaimer
This is a personal/educational project. While it provides basic fingerprint blurring, it should not be relied upon for critical privacy/security applications.
