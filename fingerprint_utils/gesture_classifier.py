import torch
import torch.nn as nn
import json
import os

class GestureCNN1D(nn.Module):
    def __init__(self, input_dim=63, num_classes=19):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        self._to_linear = None
        self._get_conv_output(input_dim)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output(self, input_dim):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_dim)
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = torch.relu(self.conv3(x))
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class GestureClassifier:
    # Finger indices: 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
    GESTURE_EXPOSED_FINGERS = {
        'peace': [8, 12],
        'peace_inverted': [],
        'ok': [12, 16, 20],
        'like': [4],
        'dislike': [4],
        'one': [8],
        'palm': [4, 8, 12, 16, 20],
        'call': [],
        'rock': [8, 20],
        'four': [8, 12, 16, 20],
        'three2': [4, 8, 12],
        'three': [8, 12, 16],
        'two_up': [8, 12],
        'two_up_inverted': [],
        'mute': [],
        'stop': [4, 8, 12, 16, 20],
        'stop_inverted': [],
        'fist': [],
    }

    def __init__(self, model_path='gesture_classifier.pth', names_path='gesture_names.json', device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(names_path, 'r') as f:
            self.gesture_names = json.load(f)

        self.model = GestureCNN1D(num_classes=len(self.gesture_names)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, features):
        """Predict gesture name from 63-dim features"""
        tensor = torch.from_numpy(features).to(self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)
            _, pred = torch.max(output, 1)
        return self.gesture_names[pred.item()]

    def get_exposed_fingers(self, features):
        """Get finger indices to protect based on predicted gesture"""
        gesture = self.predict(features)
        finger_indices = self.GESTURE_EXPOSED_FINGERS.get(gesture, [])
        print(f"Predicted gesture: {gesture}, exposed fingers: {finger_indices}")
        return finger_indices