# Pattern Recognition for Financial Time Series Forecasting
Group Project done by:
|Name|Registration Number|
|-|-|
|Archith Sunil|TCR24CS015|
|Bhagath P. R.|TCR24CS019|
|David Chacko Binoy|TCR24CS022|
|Devanandan J. Y.|TCR24CS023|
|Harinarayanan R.|TCR24CS033|
|Joseph Mathew|TCR24CS038|

**Course:** Pattern Recognition

This project treats stock price data as a signal, transforms it into a time-frequency representation using the Short-Time Fourier Transform (STFT), and trains a Convolutional Neural Network (CNN) to predict future stock prices.

## Overview

The pipeline works as follows:

```
Stock Price Data → Fourier Transform → STFT Spectrogram → CNN Model → Price Prediction
```

Three NSE-listed Indian IT companies are analyzed: **TCS**, **Infosys**, and **Wipro**, using daily closing price data from 2020 to 2024.

## Project Structure

```
assignment2
├── task1_data.py       # Download, align, and normalize stock data
├── task2_signal.py     # Compute FFT and generate STFT spectrograms
├── task3_model.py      # Build and train the CNN model
├── task4_analysis.py   # Evaluate predictions and generate plots
├── run.py              # Cross-platform launcher (handles venv setup)
├── requirements.txt    # Python dependencies
└── report.docx         # Full assignment report with plots and analysis
```

## Setup

Make sure you have Python 3.8 or later installed. Then run:

```bash
python run.py --setup
```

This will create a virtual environment and install all required packages automatically. You do not need to manually create a venv or run pip — the launcher handles everything, including on systems that protect the system Python (CachyOS, Ubuntu 22.04+).

> **Windows users:** PyTorch will be installed in CPU-only mode. Training will be slightly slower but fully functional.

> **Google Colab users:** Skip setup entirely. Just run `!pip install yfinance` in the first cell, upload the task files, and run them with `!python task1_data.py` etc.

## Running

Run the tasks in order:

```bash
python run.py task1_data.py
python run.py task2_signal.py
python run.py task3_model.py
python run.py task4_analysis.py
```

Each task depends on the output of the previous one, so the order matters.

## Output

After running all four tasks, the following will be generated:

```
assignment2/
├── data/
│   ├── raw_prices.csv
│   └── normalized_prices.csv
├── spectrograms/
│   ├── TCS.npy
│   ├── Infosys.npy
│   └── Wipro.npy
├── models/
│   ├── TCS_cnn.pt
│   ├── Infosys_cnn.pt
│   ├── Wipro_cnn.pt
│   └── all_results.npy
└── plots/
    ├── task1_time_series.png
    ├── task2_frequency_spectrum.png
    ├── task2_spectrograms.png
    ├── task3_TCS_loss.png
    ├── task3_Infosys_loss.png
    ├── task3_Wipro_loss.png
    ├── task4_predictions.png
    ├── task4_residuals.png
    ├── task4_metrics_comparison.png
    └── task4_loss_curves.png
```

## Results Summary

|Company|MSE|RMSE|MAE|R²|
|-|-|-|-|-|
|TCS|0.00563|0.07506|0.06307|0.0995|
|Infosys|0.00553|0.07437|0.06007|0.6602|
|Wipro|0.00132|0.03633|0.02931|0.8037|

All metrics are on the normalized \[0–1] price scale. See `report.pdf` for the full analysis.

## Dependencies

|Package|Purpose|
|-|-|
|yfinance|Download historical stock data|
|numpy / scipy|Signal processing and FFT|
|matplotlib|Plotting and visualization|
|torch / torchvision|CNN model (PyTorch)|
|scikit-learn|Train/val split and metrics|

## References

1. Y. Zhang and C. Aggarwal, "Stock Market Prediction Using Deep Learning," IEEE Access.
2. A. Tsantekidis et al., "Deep Learning for Financial Time Series Forecasting."
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
4. A. Borovykh et al., "Conditional Time Series Forecasting with CNNs."

