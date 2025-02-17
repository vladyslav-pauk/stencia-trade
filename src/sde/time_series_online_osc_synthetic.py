import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Load configuration from YAML
with open('config_osc.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Step 1: Generate Synthetic Data
def generate_nsde_data(timesteps, dt, noise_std, spike_length, spike_amplitude_range, spike_interval_range):
    """Generates time-series data for a noisy nonlinear oscillator."""
    t = np.arange(0, timesteps * dt, dt)
    x = np.zeros_like(t)
    v = np.zeros_like(t)

    # Initial conditions
    x[0] = config['initial_conditions']['x0']
    v[0] = config['initial_conditions']['v0']

    # Coefficients for nonlinear oscillator
    def damping_coefficient(x):
        return config['coefficients']['damping_base'] + config['coefficients']['damping_scale'] * np.sin(x)

    def stiffness_coefficient(x):
        return config['coefficients']['stiffness_base'] + config['coefficients']['stiffness_scale'] * x ** 2

    def nonlinear_term(x):
        return config['coefficients']['nonlinear_scale'] * np.tanh(x)

    def external_force(t):
        """Decaying saw-like external force with irregular timing for spikes."""
        force = np.zeros_like(t)
        noise = np.random.randn(len(t)) * config['force']['noise_scale']

        # Generate random start indices for spikes
        spike_starts = np.cumsum(np.random.randint(*spike_interval_range, size=(len(t) // 500,)))
        spike_starts = spike_starts[spike_starts < len(t)]

        for start in spike_starts:
            spike_amp = np.random.uniform(*spike_amplitude_range)
            spike = spike_amp * np.linspace(1, -0.5, spike_length)
            if (start // 50) % 2 == 0:
                spike = -spike
            force[start:start + len(spike)] += spike[:len(force) - start]

        return force + noise

    # Time evolution
    F_t = external_force(t)
    for i in range(1, len(t)):
        a_x = (
            -damping_coefficient(x[i - 1]) * v[i - 1]
            - stiffness_coefficient(x[i - 1]) * x[i - 1]
            + nonlinear_term(x[i - 1])
            + F_t[i - 1]
        )
        v[i] = v[i - 1] + a_x * dt + noise_std * np.sqrt(dt) * np.random.randn()
        x[i] = x[i - 1] + v[i - 1] * dt + noise_std * np.sqrt(dt) * np.random.randn()

    return t, x, F_t

# Dataset and DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, x, F, seq_len):
        self.x = x
        self.F = F
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx:idx + self.seq_len], dtype=torch.float32),
            torch.tensor(self.F[idx + self.seq_len], dtype=torch.float32),
            torch.tensor(self.x[idx + self.seq_len], dtype=torch.float32)
        )

# Model Definition
class CoefficientLearner(nn.Module):
    def __init__(self, seq_len):
        super(CoefficientLearner, self).__init__()
        self.seq_len = seq_len
        self.fc = nn.Sequential(
            nn.Linear(seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Damping, stiffness, nonlinear term, and F(t)
        )

    def forward(self, x):
        return self.fc(x)

# Pretraining Loop
def pretrain_model(model, dataloader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for seq, _, target in dataloader:
            optimizer.zero_grad()

            coeffs = model(seq)
            damping, stiffness, nonlinear, force = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]

            noise = torch.randn_like(seq[:, -1]) * 0.1
            predicted_next = (
                seq[:, -1]
                - damping * seq[:, -1]
                - stiffness * (seq[:, -1] ** 2)
                + nonlinear * torch.tanh(seq[:, -1])
                + force
                + noise
            )

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
        seq.requires_grad_()
        target = torch.tensor(x[i], dtype=torch.float32).unsqueeze(0)

        optimizer.zero_grad()
        coeffs = model(seq)
        damping, stiffness, nonlinear, force = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]

        noise = torch.randn(1) * 0.1
        predicted = (
            seq[0, -1]
            - damping.item() * seq[0, -1]
            - stiffness.item() * (seq[0, -1] ** 2)
            + nonlinear.item() * torch.tanh(seq[0, -1])
            + force.item()
            + noise
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
            damping, stiffness, nonlinear, force = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]

            current_seq = seq[0, -1]
            predicted_steps = []

            for _ in range(max(predict_steps)):
                noise = torch.randn(1) * 0.1
                current_seq = (
                    current_seq
                    - damping.item() * current_seq
                    - stiffness.item() * (current_seq ** 2)
                    + nonlinear.item() * torch.tanh(current_seq)
                    + force.item()
                    + noise
                )
                predicted_steps.append(current_seq.item())

            predictions.append(predicted_steps)

    predictions = np.array(predictions)[:, [s - 1 for s in predict_steps]]
    return predictions

def simple_trader(t, x, predictions, trade_interval, initial_balance=1000, transaction_cost=0):
    """
    A simple trader that buys or sells at fixed intervals based on predicted dynamics.
    """
    balance = initial_balance  # Initial cash balance
    position = 0  # Number of units held (positive for buy, negative for sell)
    trade_log = []  # To log trades and profits

    # Iterate through the predictions at fixed intervals
    for i in range(0, len(predictions) - trade_interval, trade_interval):
        current_price = x[i]
        future_price = predictions[i + trade_interval, 0]  # Predicted price after `trade_interval` steps

        if future_price > current_price:  # Buy signal
            if position <= 0:  # Only buy if not holding or shorting
                # Close any short position
                balance += position * current_price
                position = 0
                # Buy 1 unit
                position += 1
                balance -= current_price + transaction_cost
                trade_log.append((t[i], "Buy", current_price, balance))

        elif future_price < current_price:  # Sell signal
            if position >= 0:  # Only sell if not holding or longing
                # Close any long position
                balance += position * current_price
                position = 0
                # Sell 1 unit
                position -= 1
                balance += current_price - transaction_cost
                trade_log.append((t[i], "Sell", current_price, balance))

    # Close any open position at the end
    if position != 0:
        balance += position * x[-1]
        trade_log.append((t[-1], "Close Position", x[-1], balance))
        position = 0

    # Print trade log and final balance
    print(f"Final Balance: {balance}")
    for log in trade_log:
        print(f"Time: {log[0]:.2f}, Action: {log[1]}, Price: {log[2]:.2f}, Balance: {log[3]:.2f}")

    return balance, trade_log

# Visualization
def visualize_predictions_and_trades(t, x, predictions, seq_len, predict_steps, F_t, trade_log):
    """
    Unified visualization for dynamics, predictions, external force, and trading actions.
    """
    plt.figure(figsize=(12, 6), dpi=500)

    # Crop limit for plotting
    crop_limit = 1.5 * np.percentile(np.abs(x[seq_len:]), 99)

    # External force alignment
    F_t_trimmed = F_t[len(F_t) - len(x):]
    force_time = t[len(t) - len(F_t_trimmed):]
    force_scaled = F_t_trimmed * (crop_limit / np.max(np.abs(F_t_trimmed)))

    # Time range for predictions and true dynamics
    prediction_time = t[seq_len:]
    x_cropped = x[seq_len:]

    # Hard-coded colors for predictions
    prediction_colors = ["blue", "green", "cyan", "magenta"]

    # Plot true dynamics
    plt.plot(
        prediction_time,
        x_cropped,
        label="True Dynamics",
        color="red",
        linewidth=0.5
    )

    # Plot predictions
    for i, step in enumerate(predict_steps):
        prediction_values = predictions[:, i]
        valid_indices = np.abs(prediction_values) <= crop_limit  # Mask to exclude out-of-bound points
        plt.plot(
            prediction_time[:len(prediction_values)][valid_indices],  # Apply mask
            prediction_values[valid_indices],  # Filtered values
            linestyle="dashed",
            linewidth=0.5,
            color=prediction_colors[i % len(prediction_colors)],  # Cycle through colors
            label=f"Predicted ({step} Steps Ahead)"
        )

    # Plot external force
    plt.plot(
        force_time,
        force_scaled,
        label="Scaled External Force F(t)",
        color="orange",
        linestyle="dashed",
        linewidth=0.5
    )

    # Add scatter points for trading actions without duplicating legend entries
    for log in trade_log:
        time, action, price, _ = log
        color = "green" if action == "Buy" else "red"
        plt.scatter(time, price, color=color, zorder=5, label='_nolegend_')  # Exclude from legend

    # Add single legend entries for Buy and Sell
    buy_handle = plt.Line2D([], [], color="green", marker="o", linestyle="None", label="Buy")
    sell_handle = plt.Line2D([], [], color="red", marker="o", linestyle="None", label="Sell")

    # Merge handles from all plotted elements and add custom Buy/Sell
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([buy_handle, sell_handle])

    # Update legend
    plt.legend(handles=handles, loc="upper right")

    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.title("Dynamics, Predictions, External Force, and Trading Actions")
    plt.tight_layout()
    plt.savefig("unified_visualization.png")
    plt.show()


def main():
    params = config['parameters']

    # Generate data
    t, x, F_t = generate_nsde_data(
        timesteps=params['timesteps'],
        dt=params['dt'],
        noise_std=params['noise_std'],
        spike_length=config['force']['spike_length'],
        spike_amplitude_range=config['force']['spike_amplitude_range'],
        spike_interval_range=config['force']['spike_interval_range']
    )

    seq_len = params['seq_len']
    half_idx = len(x) // 2
    dataset = TimeSeriesDataset(x[:half_idx], F_t[:half_idx], seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    model = CoefficientLearner(seq_len=seq_len)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()

    # Pretraining
    print("Starting pretraining...")
    pretrain_model(model, dataloader, optimizer, criterion, epochs=params['epochs'])

    # Online learning
    print("Starting online learning...")
    online_learning(model, x[half_idx:], seq_len, optimizer, criterion)

    # Predictions
    predict_steps = params.get('predict_steps', [1, 10, 50, 100])
    predictions = test_model(model, x[half_idx:], seq_len, predict_steps)

    # Trading
    final_balance, trade_log = simple_trader(
        t[half_idx:],
        x[half_idx:],
        predictions,
        trade_interval=10,  # Trade every 10 steps
        initial_balance=1000,
        transaction_cost=1
    )

    # Unified Visualization
    visualize_predictions_and_trades(
        t[half_idx:], x[half_idx:], predictions, seq_len, predict_steps, F_t, trade_log
    )


if __name__ == "__main__":
    main()