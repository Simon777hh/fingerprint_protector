import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ========== Config ==========
DATA_ROOT = r"C:\Users\a\PycharmProjects\fingerprint\data\hagrid-sample-120k-384p\hagrid_120k"
MAX_PER_CLASS = 700
OUTPUT_FILE = "gesture_data.npz"

GESTURE_NAMES = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
    'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted',
    'three', 'three2', 'two_up', 'two_up_inverted'
]
GESTURE_TO_IDX = {name: i for i, name in enumerate(GESTURE_NAMES)}

# ========== Initialize MediaPipe ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


def extract_features(image_path):
    """Extract 63-dim features, return numpy array or None"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    features = []
    for lm in hand.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features, dtype=np.float32)


def load_and_save_data():
    """Load all data and save to npz file"""
    X, y = [], []

    for gesture in GESTURE_NAMES:
        folder = os.path.join(DATA_ROOT, f'train_val_{gesture}')
        if not os.path.exists(folder):
            print(f"Skip missing folder: {folder}")
            continue

        images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
        if len(images) > MAX_PER_CLASS:
            images = images[:MAX_PER_CLASS]

        label = GESTURE_TO_IDX[gesture]
        print(f"\nLoading {gesture}: {len(images)} images")

        for img_file in tqdm(images, desc=gesture):
            img_path = os.path.join(folder, img_file)
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label)

    if hands is not None:
        hands.close()

    X = np.array(X)
    y = np.array(y)

    np.savez(OUTPUT_FILE, X=X, y=y, gesture_names=GESTURE_NAMES)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    load_and_save_data()