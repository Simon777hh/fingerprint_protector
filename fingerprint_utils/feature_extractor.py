import cv2
import numpy as np
import mediapipe as mp

class HandFeatureExtractor:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence
        )

    def extract_features(self, image_path):
        """Extract 63-dim features from image file"""
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        return self.extract_features_from_array(img)

    def extract_features_from_array(self, image):
        """Extract 63-dim features from image array (BGR)"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None, None
        hand = results.multi_hand_landmarks[0]
        features = []
        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])
        return np.array(features, dtype=np.float32), hand

    def get_hand_landmarks(self, image):
        """Get raw hand landmarks from image array"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        return results.multi_hand_landmarks[0]

    def close(self):
        if self.hands is not None:
            self.hands.close()