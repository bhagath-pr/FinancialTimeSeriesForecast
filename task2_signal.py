import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
import os

os.makedirs("spectrograms", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Load Normalized Data ───────────────────────────────────────────────────────

print("Loading normalized data...")
df = pd.read_csv("data/normalized_prices.csv", index_col=0, parse_dates=True)
companies = df.columns.tolist()
print(f"  Companies: {companies}")
print(f"  Days: {len(df)}")

# ── STFT Parameters ────────────────────────────────────────────────────────────

# Window length: how many days per segment
# 8 days ≈ 2 weeks — short window gives better time resolution
# and produces far more spectrogram time steps (= more training samples)
WINDOW_LENGTH = 8

# Hop size: 1 day — slide by 1 day to maximise number of samples
HOP_SIZE = 1

# ── Step 1: Fourier Transform (full signal) ────────────────────────────────────
# This shows which "frequencies" (periodicities) dominate each stock overall

print("\nComputing Fourier Transforms...")

fig, axes = plt.subplots(len(companies), 1, figsize=(12, 4 * len(companies)))

for i, company in enumerate(companies):
    signal = df[company].values

    # Compute FFT
    fft_vals  = np.fft.rfft(signal)           # real-input FFT (no redundant mirror)
    fft_mag   = np.abs(fft_vals)              # magnitude spectrum
    fft_freqs = np.fft.rfftfreq(len(signal))  # frequencies (in cycles/day)

    # Convert frequency to period in trading days for easier interpretation
    # Avoid division by zero for the DC component (freq=0)
    with np.errstate(divide="ignore"):
        periods = np.where(fft_freqs > 0, 1.0 / fft_freqs, np.inf)

    axes[i].plot(periods[1:], fft_mag[1:], color="steelblue")  # skip DC component
    axes[i].set_title(f"{company} — Frequency Spectrum")
    axes[i].set_xlabel("Period (trading days)")
    axes[i].set_ylabel("Magnitude")
    axes[i].set_xlim(0, 260)   # show up to ~1 year period
    axes[i].grid(True, alpha=0.3)

    # Annotate dominant period
    dominant_idx    = np.argmax(fft_mag[1:]) + 1
    dominant_period = periods[dominant_idx]
    axes[i].axvline(dominant_period, color="red", linestyle="--", alpha=0.7,
                    label=f"Dominant period: {dominant_period:.0f} days")
    axes[i].legend()

    print(f"  {company}: dominant period = {dominant_period:.0f} trading days")

plt.tight_layout()
plt.savefig("plots/task2_frequency_spectrum.png", dpi=150)
plt.show()
print("Saved: plots/task2_frequency_spectrum.png")

# ── Step 2: STFT Spectrogram ───────────────────────────────────────────────────
# Unlike FFT, STFT shows how frequency content CHANGES over time

print("\nComputing STFT Spectrograms...")

# We'll store spectrograms for use in Task 3
all_spectrograms = {}

fig, axes = plt.subplots(len(companies), 1, figsize=(14, 5 * len(companies)))

for i, company in enumerate(companies):
    signal = df[company].values

    # scipy.signal.stft returns:
    #   f      → frequency bins
    #   t      → time segments (in samples)
    #   Zxx    → complex STFT matrix, shape (freq_bins, time_steps)
    f, t, Zxx = stft(signal,
                     nperseg=WINDOW_LENGTH,   # window length
                     noverlap=WINDOW_LENGTH - HOP_SIZE)  # overlap = L - H

    # Spectrogram = magnitude squared of STFT
    spectrogram = np.abs(Zxx) ** 2

    # Save for Task 3
    all_spectrograms[company] = spectrogram

    # Plot
    ax = axes[i]
    img = ax.pcolormesh(t, f, spectrogram, shading="gouraud", cmap="inferno")
    ax.set_title(f"{company} — Spectrogram (STFT)")
    ax.set_xlabel("Time (trading days)")
    ax.set_ylabel("Frequency (cycles/day)")
    fig.colorbar(img, ax=ax, label="Energy")

    print(f"  {company}: spectrogram shape = {spectrogram.shape}  "
          f"(freq_bins × time_steps)")

plt.tight_layout()
plt.savefig("plots/task2_spectrograms.png", dpi=150)
plt.show()
print("Saved: plots/task2_spectrograms.png")

# ── Step 3: Save Spectrograms ──────────────────────────────────────────────────
# Save as .npy files so Task 3 can load them directly without recomputing

print("\nSaving spectrograms...")
for company, spec in all_spectrograms.items():
    path = f"spectrograms/{company}.npy"
    np.save(path, spec)
    print(f"  Saved: {path}  shape={spec.shape}")

print("\nTask 2 complete!")
