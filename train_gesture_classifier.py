import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ========== Config ==========
DATA_FILE = "gesture_data.npz"
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 5e-4

# ========== Load Data ==========
data = np.load(DATA_FILE, allow_pickle=True)
X = data['X']
y = data['y']
GESTURE_NAMES = data['gesture_names'].tolist()
NUM_CLASSES = len(GESTURE_NAMES)

print(f"Loaded {len(X)} samples")
print(f"Feature shape: {X.shape}")
print(f"Classes: {NUM_CLASSES}")

# ========== Split Data ==========
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== Convert to Tensors ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# ========== Model Definition (1D-CNN) ==========
class GestureCNN1D(nn.Module):
    def __init__(self, input_dim=63, num_classes=NUM_CLASSES):
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


# ========== Training ==========
model = GestureCNN1D().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            output = model(batch_X)
            pred = torch.argmax(output, 1)
            correct += (pred == batch_y).sum().item()

    acc = correct / len(val_dataset)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}, Val Acc: {acc:.4f}")

# ========== Save Model ==========
torch.save(model.state_dict(), 'gesture_classifier.pth')
print("\nModel saved to gesture_classifier.pth")

import json

with open('gesture_names.json', 'w') as f:
    json.dump(GESTURE_NAMES, f)
print("Gesture names saved to gesture_names.json")