import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Step 1: Generate Synthetic Data
def generate_nsde_data(timesteps=5000, dt=0.01, noise_std=0.5):
    """Generates time-series data for a noisy nonlinear oscillator."""
    t = np.arange(0, timesteps * dt, dt)
    x = np.zeros_like(t)
    v = np.zeros_like(t)

    # Initial conditions
    x[0] = 1.0
    v[0] = 0.0

    # Coefficients for nonlinear oscillator
    def damping_coefficient(x):
        return 0.1 + 0.05 * np.sin(x)

    def stiffness_coefficient(x):
        return 1.0 + 0.1 * x ** 2

    # Time evolution
    for i in range(1, len(t)):
        a_x = -damping_coefficient(x[i - 1]) * v[i - 1] - stiffness_coefficient(x[i - 1]) * x[i - 1]
        v[i] = v[i - 1] + a_x * dt + noise_std * np.sqrt(dt) * np.random.randn()
        x[i] = x[i - 1] + v[i - 1] * dt

    return t, x, v

# Step 2: Dataset and DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, t, x, seq_len=10):
        self.x = x
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx:idx + self.seq_len], dtype=torch.float32),
            torch.tensor(self.x[idx + self.seq_len], dtype=torch.float32)
        )

# Step 3: Define the Model
class CoefficientLearner(nn.Module):
    def __init__(self, seq_len):
        super(CoefficientLearner, self).__init__()
        self.seq_len = seq_len
        self.fc = nn.Sequential(
            nn.Linear(seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Two coefficients: damping and stiffness
        )

    def forward(self, x):
        return self.fc(x)

# Step 4: Training Loop
def train_model(model, dataloader, optimizer, criterion, epochs=20):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for seq, target in dataloader:
            optimizer.zero_grad()

            # Predict coefficients
            coeffs = model(seq)
            damping, stiffness = coeffs[:, 0], coeffs[:, 1]

            # Predict next step
            predicted_next = seq[:, -1] - damping * seq[:, -1] - stiffness * (seq[:, -1] ** 2)

            loss = criterion(predicted_next, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# Step 5: Testing and Visualization
def test_model(model, t, x, seq_len):
    model.eval()
    predicted = []
    with torch.no_grad():
        for i in range(seq_len, len(x)):
            seq = torch.tensor(x[i - seq_len:i], dtype=torch.float32).unsqueeze(0)
            coeffs = model(seq)
            damping, stiffness = coeffs[:, 0], coeffs[:, 1]
            predicted_next = seq[0, -1] - damping * seq[0, -1] - stiffness * (seq[0, -1] ** 2)
            predicted.append(predicted_next.item())

    return np.array(predicted)

# Generate data
t, x, _ = generate_nsde_data()

# Prepare dataset and dataloader
seq_len = 100
dataset = TimeSeriesDataset(t, x, seq_len=seq_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, optimizer, and loss function
model = CoefficientLearner(seq_len=seq_len)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Train the model
train_model(model, dataloader, optimizer, criterion, epochs=50)

# Test and visualize predictions
predicted = test_model(model, t, x, seq_len)

plt.figure(figsize=(10, 6))
plt.plot(t[seq_len:], x[seq_len:], label='True Dynamics')
plt.plot(t[seq_len:], predicted, label='Predicted Dynamics', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.legend()
plt.title('True vs Predicted Dynamics')
plt.show()
