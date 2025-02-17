import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yfinance as yf


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


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input):
        if input.size(0) == 1:
            return input
        else:
            return super(SafeBatchNorm1d, self).forward(input)


class GBMLearner(nn.Module):
    def __init__(self, seq_len):
        super(GBMLearner, self).__init__()
        self.seq_len = seq_len
        self.fc = nn.Sequential(
            nn.Linear(seq_len, 512),
            SafeBatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            SafeBatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            SafeBatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


def fetch_data(stock_ticker, sp_ticker, start_date, end_date):
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
    sp_data = yf.download(sp_ticker, start=start_date, end=end_date)

    stock_prices = stock_data['Adj Close'].dropna()
    sp_prices = sp_data['Adj Close'].dropna()

    stock_prices = stock_prices / stock_prices.iloc[0]
    sp_prices = sp_prices / sp_prices.iloc[0]

    return stock_prices.values, sp_prices.values


def pretrain_model(model, dataloader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for seq, target in dataloader:
            optimizer.zero_grad()
            coeffs = model(seq)
            mu, sigma = coeffs[:, 0], coeffs[:, 1]
            noise = torch.randn_like(target) * 0.0001
            predicted_next = (
                    seq[:, -1].squeeze() * torch.exp((mu - 0.5 * sigma ** 2) + sigma * noise.squeeze())
            ).unsqueeze(-1)
            loss = criterion(predicted_next, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Pretraining Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")


def online_learning_and_prediction(model, x, seq_len, optimizer, criterion, predict_steps):
    """
    Generate predictions for multiple time steps ahead based on given predict_steps.
    """
    model.eval()
    predictions_dict = {step: [] for step in predict_steps}  # Dictionary to store predictions for each step

    for i in range(seq_len, len(x)):
        seq = torch.tensor(x[i - seq_len:i], dtype=torch.float32).unsqueeze(0)
        coeffs = model(seq)
        mu, sigma = coeffs[:, 0], coeffs[:, 1]

        # Generate predictions for each step in predict_steps
        for step in predict_steps:
            current_seq = x[i - 1]  # Start from the last value in the sequence
            for _ in range(step):
                noise = np.random.randn() * 0.0001
                current_seq = current_seq * np.exp((mu.item() - 0.5 * sigma.item() ** 2) + sigma.item() * noise)
            predictions_dict[step].append(current_seq)

    # Convert lists to numpy arrays
    for step in predict_steps:
        predictions_dict[step] = np.array(predictions_dict[step])

    return predictions_dict


def simple_trader(t, x, predictions, trade_interval, initial_balance=1000, transaction_cost=0.001,
                    stop_loss=0.02, take_profit=0.05):
    """
    Enhanced trading strategy with stop-loss, take-profit, and dynamic position sizing.

    Parameters:
        t (np.ndarray): Time array corresponding to prices.
        x (np.ndarray): Array of true price values.
        predictions (np.ndarray): Predicted price values.
        trade_interval (int): Number of steps ahead to base decisions on.
        initial_balance (float): Starting account balance.
        transaction_cost (float): Cost per trade as a fraction of trade size.
        stop_loss (float): Stop-loss threshold as a fraction of trade entry price.
        take_profit (float): Take-profit threshold as a fraction of trade entry price.

    Returns:
        trade_performance (np.ndarray): Portfolio value over time.
        trade_log (list): Log of all trades executed.
    """
    balance = [initial_balance]
    position = [0]  # Position size (e.g., number of shares held)
    entry_price = 0  # Price at which the current position was entered
    trade_log = []
    trade_performance = np.zeros(len(x))

    for i in range(len(predictions)):
        current_price = x[i]

        # Update portfolio value
        trade_performance[i] = balance[0] + position[0] * current_price[0]

        # Check for stop-loss or take-profit conditions
        # Check for stop-loss or take-profit conditions
        if position != 0 and entry_price != 0:  # Ensure entry_price is non-zero
            price_change = (current_price - entry_price) / entry_price
            if price_change <= -stop_loss:
                # Stop-loss triggered
                balance += position * current_price * (1 - transaction_cost)
                trade_log.append((t[i], "Stop-Loss", current_price, balance))
                position = [0]
                entry_price = 0
                continue
            elif price_change >= take_profit:
                # Take-profit triggered
                balance += position * current_price * (1 - transaction_cost)
                trade_log.append((t[i], "Take-Profit", current_price, balance))
                position = [0]
                entry_price = 0
                continue

        # Execute trades based on predictions
        if i >= trade_interval:
            future_price = predictions[i - trade_interval]

            # Buy signal: Predicted price > current price
            if future_price > current_price[0] and position[0] <= 0:
                # Close short position if any
                if position[0] < 0:
                    balance += position * current_price * (1 - transaction_cost)
                # Calculate position size dynamically (e.g., fraction of balance)
                position_size = balance[0] * 0.1 / current_price  # Risking 10% of balance
                balance -= position_size * current_price * (1 + transaction_cost)
                position = position_size
                entry_price = current_price
                trade_log.append((t[i], "Buy", current_price, balance))

            # Sell signal: Predicted price < current price
            elif future_price < current_price[0] and position[0] >= 0:
                # Close long position if any
                if position[0] > 0:
                    balance += position * current_price * (1 - transaction_cost)
                # Calculate position size dynamically
                position_size = balance[0] * 0.1 / current_price  # Risking 10% of balance
                balance -= position_size * current_price * (1 + transaction_cost)
                position = -position_size
                entry_price = current_price
                trade_log.append((t[i], "Sell", current_price, balance))

    # Close any open position at the end
    if position != 0:
        balance += position * x[-1] * (1 - transaction_cost)
        trade_log.append((t[-1], "Close Position", x[-1], balance))
        trade_performance[-1] = balance[0]

    print(f"Final Balance: {balance[0]:.2f}")
    return trade_performance, trade_log


def plot_trade_vs_sp(t, trade_performance, sp_prices, initial_balance, seq_len):
    sp_normalized = sp_prices * (initial_balance / sp_prices[0])

    plt.figure(figsize=(6, 3), dpi=500)
    plt.plot(t[:-seq_len], trade_performance[:-seq_len], label="Trade Strategy", color="blue")
    plt.plot(t[:-seq_len], sp_normalized[seq_len:len(t)+seq_len], label="S&P 500", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Trade Strategy vs S&P 500 Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_predictions(t, true_values, predictions_dict, seq_len):
    """
    Visualize true values and predictions as seen by the loss function.

    Parameters:
        t (np.ndarray): Time array.
        true_values (np.ndarray): True price values (raw, unnormalized).
        predictions_dict (dict): Predicted values for multiple steps.
        seq_len (int): Sequence length used for predictions.
    """
    adjusted_t = t[seq_len:]  # Align time with sequence length

    plt.figure(figsize=(8, 4), dpi=500)
    plt.plot(adjusted_t, true_values[seq_len:], label="True Prices", color="blue")

    # Plot predictions as passed to the loss function
    for step_size, predictions in predictions_dict.items():
        aligned_t = adjusted_t[:len(predictions)]
        plt.plot(aligned_t, predictions, label=f"{step_size}-Step Prediction", linestyle="--")

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("True Prices vs Predictions for Multiple Steps (Raw Values)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    stock_ticker = "AAPL"
    sp_ticker = "^GSPC"
    start_date = "2013-01-01"
    end_date = "2023-01-01"
    seq_len = 30
    predict_steps = [1, 5, 10, 20, 50]  # Predicting for different steps

    stock_prices, sp_prices = fetch_data(stock_ticker, sp_ticker, start_date, end_date)
    half_idx = len(stock_prices) // 2
    pretrain_data = stock_prices[:half_idx]
    online_data = stock_prices[half_idx:]

    dataset = TimeSeriesDataset(pretrain_data, seq_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GBMLearner(seq_len=seq_len)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    print("Starting pretraining...")
    pretrain_model(model, dataloader, optimizer, criterion, epochs=300)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print("Starting online learning...")
    predictions_dict = online_learning_and_prediction(model, online_data, seq_len, optimizer, criterion, predict_steps)

    print("Starting trading...")
    t = np.arange(len(online_data))
    for step, predictions in predictions_dict.items():
        trade_performance, trade_log = simple_trader(t, online_data, predictions, trade_interval=step)
        plot_trade_vs_sp(t, trade_performance, sp_prices[half_idx:], initial_balance=1000, seq_len=seq_len)

    visualize_predictions(t, online_data, predictions_dict, seq_len)

if __name__ == "__main__":
    main()