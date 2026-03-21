import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────

COMPANIES     = ["TCS", "Infosys", "Wipro"]
PREDICT_AHEAD = 5        # predict price 5 trading days into the future
BATCH_SIZE    = 32
EPOCHS        = 100
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load Data ──────────────────────────────────────────────────────────────────

print("\nLoading data...")
prices_df = pd.read_csv("data/normalized_prices.csv", index_col=0, parse_dates=True)

# ── Dataset ────────────────────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """
    Each sample is:
      X → one spectrogram column (snapshot of frequency content at time t)
      y → normalized stock price at time t + PREDICT_AHEAD

    The spectrogram has shape (freq_bins, time_steps).
    Each column (time step) is a 1D vector of frequency energies.
    We treat it as a single-channel 2D image: (1, freq_bins, 1) ... actually
    we stack a small context window of columns to give the CNN spatial structure.
    """
    def __init__(self, spectrogram, prices, predict_ahead, context=8):
        self.samples = []
        self.labels  = []

        n_time = spectrogram.shape[1]

        for t in range(context, n_time):
            # Context window: last `context` columns of spectrogram up to time t
            # Shape: (freq_bins, context)
            spec_window = spectrogram[:, t - context: t]

            # Add channel dimension → (1, freq_bins, context)
            spec_window = spec_window[np.newaxis, :, :]

            # Target: price `predict_ahead` steps after current time t
            price_idx = t + predict_ahead
            if price_idx >= len(prices):
                break

            self.samples.append(spec_window.astype(np.float32))
            self.labels.append(np.float32(prices[price_idx]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx]),
                torch.tensor(self.labels[idx]))

# ── CNN Model ──────────────────────────────────────────────────────────────────

class StockCNN(nn.Module):
    """
    A simple CNN that takes a spectrogram window as input and
    predicts a future stock price (regression).

    Architecture:
      Conv2d → ReLU → MaxPool
      Conv2d → ReLU → MaxPool
      Flatten
      Linear → ReLU → Dropout
      Linear (output: 1 value)
    """
    def __init__(self, freq_bins, context):
        super(StockCNN, self).__init__()

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, padding=1),  # keeps spatial size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),           # halves spatial dims

            # Block 2
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Calculate flattened size after conv blocks
        dummy = torch.zeros(1, 1, freq_bins, context)
        conv_out = self.conv_block(dummy)
        flat_size = conv_out.view(1, -1).shape[1]

        self.fc_block = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),        # reduces overfitting
            nn.Linear(64, 1),       # single output: predicted price
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_block(x)
        return x.squeeze(1)         # shape: (batch,)

# ── Training Loop ──────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses   = []

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)
                val_batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)
        val_loss   = np.mean(val_batch_losses)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} — "
                  f"Train MSE: {train_loss:.5f}  Val MSE: {val_loss:.5f}")

    return train_losses, val_losses

# ── Main: Train One Model Per Company ─────────────────────────────────────────

CONTEXT = 16  # number of spectrogram columns used as input window

all_results = {}   # store predictions for Task 4

for company in COMPANIES:
    print(f"\n{'='*50}")
    print(f"  Training model for {company}")
    print(f"{'='*50}")

    # Load spectrogram
    spec = np.load(f"spectrograms/{company}.npy")   # shape: (freq_bins, time_steps)
    freq_bins = spec.shape[0]

    # Load prices aligned to spectrogram time steps
    prices = prices_df[company].values

    # Build dataset
    dataset    = SpectrogramDataset(spec, prices, PREDICT_AHEAD, CONTEXT)
    train_data, val_data = train_test_split(dataset, test_size=0.2, shuffle=False)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Dataset: {len(dataset)} samples  "
          f"({len(train_data)} train / {len(val_data)} val)")

    # Build model
    model = StockCNN(freq_bins=freq_bins, context=CONTEXT).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Train
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, EPOCHS, LEARNING_RATE)

    # Save model
    model_path = f"models/{company}_cnn.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  Saved model: {model_path}")

    # ── Plot Training Curves ──
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses,   label="Val MSE")
    plt.title(f"{company} — Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/task3_{company}_loss.png", dpi=150)
    plt.close()

    # ── Collect Predictions on Validation Set ──
    model.eval()
    all_preds  = []
    all_actual = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds   = model(X_batch).cpu().numpy()
            all_preds.extend(preds)
            all_actual.extend(y_batch.numpy())

    all_results[company] = {
        "actual":       np.array(all_actual),
        "predicted":    np.array(all_preds),
        "train_losses": train_losses,
        "val_losses":   val_losses,
    }

# Save results for Task 4
np.save("models/all_results.npy", all_results)
print("\nSaved predictions: models/all_results.npy")
print("\nTask 3 complete!")
