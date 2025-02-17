import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Load configuration from YAML
with open('config_jdm.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Step 1: Generate Jump-Diffusion Synthetic Data
def generate_jump_diffusion_data(timesteps, dt, mu, sigma, S0, jump_intensity, jump_mean, jump_std):
    """
    Generates time-series data for a Jump-Diffusion process.

    Args:
        timesteps (int): Number of timesteps to simulate.
        dt (float): Time increment.
        mu (float): Drift coefficient.
        sigma (float): Volatility coefficient.
        S0 (float): Initial price.
        jump_intensity (float): Poisson intensity (average number of jumps per unit time).
        jump_mean (float): Mean of the jump size (log-space).
        jump_std (float): Standard deviation of the jump size (log-space).

    Returns:
        t (np.ndarray): Time array.
        S (np.ndarray): Simulated price process.
    """
    t = np.arange(0, timesteps * dt, dt)
    S = np.zeros_like(t)
    S[0] = S0

    for i in range(1, len(t)):
        dW = np.sqrt(dt) * np.random.randn()
        jump_occurred = np.random.poisson(jump_intensity * dt) > 0
        jump_size = np.random.normal(jump_mean, jump_std) if jump_occurred else 0

        S[i] = S[i - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * dW + jump_size
        )

    return t, S

# Dataset and DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, x, seq_len):
        self.x = x
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx:idx + self.seq_len], dtype=torch.float32),
            torch.tensor(self.x[idx + self.seq_len], dtype=torch.float32)
        )

# Model Definition
class JumpDiffusionLearner(nn.Module):
    def __init__(self, seq_len):
        super(JumpDiffusionLearner, self).__init__()
        self.seq_len = seq_len
        self.fc = nn.Sequential(
            nn.Linear(seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Drift (mu), Volatility (sigma), Jump Intensity (lambda), Jump Mean (mu_j), Jump Std (sigma_j)
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        output = self.fc(x)
        mu = output[:, 0]
        sigma = self.softplus(output[:, 1])  # Ensure volatility (sigma) is non-negative
        jump_intensity = self.softplus(output[:, 2])  # Ensure jump intensity (lambda) is non-negative
        jump_intensity = torch.clamp(jump_intensity, min=1e-5, max=10.0)  # Prevent very small or large values
        jump_mean = output[:, 3]
        jump_std = self.softplus(output[:, 4])  # Ensure jump standard deviation is non-negative
        return mu, sigma, jump_intensity, jump_mean, jump_std

# Pretraining Loop
def pretrain_model(model, dataloader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for seq, target in dataloader:
            optimizer.zero_grad()

            coeffs = model(seq)
            mu, sigma, jump_intensity, jump_mean, jump_std = coeffs

            # Ensure non-negative jump intensity
            jump_intensity = torch.clamp(jump_intensity, min=1e-5)

            # Predict next value
            noise = torch.randn_like(target) * torch.sqrt(torch.tensor(config['parameters']['dt']))
            jumps = torch.poisson(jump_intensity * config['parameters']['dt']) * (
                jump_mean + jump_std * torch.randn_like(target)
            )
            predicted_next = seq[:, -1] * torch.exp((mu - 0.5 * sigma ** 2) * config['parameters']['dt'] + sigma * noise) + jumps

            loss = criterion(predicted_next, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Pretraining Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# Online Learning
def online_learning(model, x, seq_len, optimizer, criterion):
    model.train()
    total_loss = 0

    for i in range(seq_len, len(x) - 1):
        seq = torch.tensor(x[i - seq_len:i], dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(x[i], dtype=torch.float32).unsqueeze(0)

        optimizer.zero_grad()
        coeffs = model(seq)
        mu, sigma, jump_intensity, jump_mean, jump_std = coeffs

        noise = torch.randn(1) * np.sqrt(config['parameters']['dt'])
        jump_occurred = torch.bernoulli(jump_intensity * config['parameters']['dt'])
        jump_size = jump_occurred * torch.normal(jump_mean, jump_std)

        predicted = seq[0, -1] * torch.exp(
            (mu.item() - 0.5 * sigma.item() ** 2) * config['parameters']['dt'] + sigma.item() * noise + jump_size
        )

        loss = criterion(predicted, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i - seq_len) % 50 == 0:
            print(f"Step {i - seq_len + 1}/{len(x) - seq_len}, Loss: {loss.item():.4f}")

    print(f"Total Online Learning Loss: {total_loss / (len(x) - seq_len):.4f}")

# Multi-Step Prediction
def test_model(model, x, seq_len, predict_steps):
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(seq_len, len(x) - max(predict_steps)):
            seq = torch.tensor(x[i - seq_len:i], dtype=torch.float32).unsqueeze(0)

            coeffs = model(seq)
            mu, sigma, jump_intensity, jump_mean, jump_std = coeffs

            current_seq = seq[0, -1]
            predicted_steps = []

            for _ in range(max(predict_steps)):
                noise = torch.randn(1) * np.sqrt(config['parameters']['dt'])
                jump_occurred = torch.bernoulli(jump_intensity * config['parameters']['dt'])
                jump_size = jump_occurred * torch.normal(jump_mean, jump_std)
                current_seq = current_seq * torch.exp(
                    (mu.item() - 0.5 * sigma.item() ** 2) * config['parameters']['dt'] + sigma.item() * noise + jump_size
                )
                predicted_steps.append(current_seq.item())

            predictions.append(predicted_steps)

    predictions = np.array(predictions)[:, [s - 1 for s in predict_steps]]
    return predictions

# Visualization
def visualize_jump_diffusion(t, S, predictions, seq_len, predict_steps):
    plt.figure(figsize=(10, 5))
    plt.plot(t, S, label='True Jump-Diffusion', color='blue')

    for i, step in enumerate(predict_steps):
        plt.plot(
            t[seq_len:seq_len + len(predictions[:, i])],
            predictions[:, i],
            label=f'Prediction ({step} Steps Ahead)',
            linestyle='dashed'
        )

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Jump-Diffusion Predictions')
    plt.tight_layout()
    plt.show()

# Main Function
def main():
    params = config['parameters']

    # Generate data
    t, S = generate_jump_diffusion_data(
        timesteps=params['timesteps'],
        dt=params['dt'],
        mu=params['mu'],
        sigma=params['sigma'],
        S0=params['S0'],
        jump_intensity=params['jump_intensity'],
        jump_mean=params['jump_mean'],
        jump_std=params['jump_std']
    )

    seq_len = params['seq_len']
    half_idx = len(S) // 2
    dataset = TimeSeriesDataset(S[:half_idx], seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    model = JumpDiffusionLearner(seq_len=seq_len)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()

    # Pretraining
    print("Starting pretraining...")
    pretrain_model(model, dataloader, optimizer, criterion, epochs=params['epochs'])

    # Online learning
    print("Starting online learning...")
    online_learning(model, S[half_idx:], seq_len, optimizer, criterion)

    # Predictions
    predict_steps = params.get('predict_steps', [1, 10, 50, 100])
    predictions = test_model(model, S[half_idx:], seq_len, predict_steps)

    # Visualization
    visualize_jump_diffusion(t[half_idx:], S[half_idx:], predictions, seq_len, predict_steps)

if __name__ == "__main__":
    main()