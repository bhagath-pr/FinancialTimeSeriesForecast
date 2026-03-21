import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

os.makedirs("plots", exist_ok=True)

COMPANIES = ["TCS", "Infosys", "Wipro"]

# ── Load Results from Task 3 ───────────────────────────────────────────────────

print("Loading predictions from Task 3...")
all_results = np.load("models/all_results.npy", allow_pickle=True).item()

# ── Step 1: Compute Metrics Per Company ───────────────────────────────────────

print("\n" + "="*55)
print(f"  {'Company':<12} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
print("="*55)

metrics = {}

for company in COMPANIES:
    actual    = all_results[company]["actual"]
    predicted = all_results[company]["predicted"]

    mse  = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(actual, predicted)
    r2   = r2_score(actual, predicted)

    metrics[company] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    print(f"  {company:<12} {mse:>10.5f} {rmse:>10.5f} {mae:>10.5f} {r2:>8.4f}")

print("="*55)
print("  (All values are on the normalized [0–1] scale)")

# ── Step 2: Prediction vs Actual Plot ─────────────────────────────────────────

print("\nGenerating prediction plots...")

fig, axes = plt.subplots(len(COMPANIES), 1, figsize=(14, 5 * len(COMPANIES)))

for i, company in enumerate(COMPANIES):
    actual    = all_results[company]["actual"]
    predicted = all_results[company]["predicted"]

    ax = axes[i]
    ax.plot(actual,    label="Actual",    color="steelblue", linewidth=1.5)
    ax.plot(predicted, label="Predicted", color="tomato",
            linewidth=1.5, linestyle="--")

    mse = metrics[company]["MSE"]
    r2  = metrics[company]["R2"]
    ax.set_title(f"{company} — Actual vs Predicted  "
                 f"(MSE={mse:.5f}, R²={r2:.4f})")
    ax.set_xlabel("Validation Sample")
    ax.set_ylabel("Normalized Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/task4_predictions.png", dpi=150)
plt.show()
print("Saved: plots/task4_predictions.png")

# ── Step 3: Residual Plot ──────────────────────────────────────────────────────
# Residual = actual - predicted
# A good model has residuals randomly scattered around 0

print("Generating residual plots...")

fig, axes = plt.subplots(len(COMPANIES), 1, figsize=(14, 4 * len(COMPANIES)))

for i, company in enumerate(COMPANIES):
    actual    = all_results[company]["actual"]
    predicted = all_results[company]["predicted"]
    residuals = actual - predicted

    ax = axes[i]
    ax.plot(residuals, color="darkorange", linewidth=1, label="Residual")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(range(len(residuals)), residuals, 0,
                    alpha=0.2, color="darkorange")
    ax.set_title(f"{company} — Residuals (Actual − Predicted)")
    ax.set_xlabel("Validation Sample")
    ax.set_ylabel("Residual")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/task4_residuals.png", dpi=150)
plt.show()
print("Saved: plots/task4_residuals.png")

# ── Step 4: Metrics Comparison Bar Chart ──────────────────────────────────────

print("Generating metrics comparison chart...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

metric_names = ["MSE", "RMSE", "MAE"]
colors       = ["steelblue", "seagreen", "tomato"]

for j, (metric, color) in enumerate(zip(metric_names, colors)):
    values = [metrics[c][metric] for c in COMPANIES]
    bars   = axes[j].bar(COMPANIES, values, color=color, alpha=0.8, edgecolor="black")
    axes[j].set_title(f"{metric} by Company")
    axes[j].set_ylabel(metric)
    axes[j].grid(True, alpha=0.3, axis="y")

    # Annotate bars with values
    for bar, val in zip(bars, values):
        axes[j].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.0005,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)

plt.suptitle("Model Performance Comparison Across Companies", fontsize=13)
plt.tight_layout()
plt.savefig("plots/task4_metrics_comparison.png", dpi=150)
plt.show()
print("Saved: plots/task4_metrics_comparison.png")

# ── Step 5: Training Loss Curves (all companies, one plot) ────────────────────

print("Generating combined loss curves...")

fig, axes = plt.subplots(1, len(COMPANIES), figsize=(15, 4))

for i, company in enumerate(COMPANIES):
    train_losses = all_results[company]["train_losses"]
    val_losses   = all_results[company]["val_losses"]

    axes[i].plot(train_losses, label="Train MSE", color="steelblue")
    axes[i].plot(val_losses,   label="Val MSE",   color="tomato", linestyle="--")
    axes[i].set_title(f"{company} — Loss Curve")
    axes[i].set_xlabel("Epoch")
    axes[i].set_ylabel("MSE Loss")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/task4_loss_curves.png", dpi=150)
plt.show()
print("Saved: plots/task4_loss_curves.png")

# ── Step 6: Summary Analysis ──────────────────────────────────────────────────

print("\n" + "="*55)
print("  ANALYSIS SUMMARY")
print("="*55)

best_mse     = min(COMPANIES, key=lambda c: metrics[c]["MSE"])
best_r2      = max(COMPANIES, key=lambda c: metrics[c]["R2"])
hardest      = max(COMPANIES, key=lambda c: metrics[c]["MSE"])

print(f"\n  Best MSE  : {best_mse} ({metrics[best_mse]['MSE']:.5f})")
print(f"  Best R²   : {best_r2}  ({metrics[best_r2]['R2']:.4f})")
print(f"  Hardest to predict: {hardest} ({metrics[hardest]['MSE']:.5f})")

print("""
  Observations:
  ─────────────────────────────────────────────────────
  • Lower MSE/RMSE/MAE → predictions are closer to actual
  • R² closer to 1.0   → model explains more variance
  • Residuals near 0   → systematic errors are small
  • If val loss >> train loss → model may be overfitting
    (fix: more dropout, fewer epochs, more data)
  • High-frequency components in spectrogram capture
    short-term volatility; low-frequency components
    capture longer trends — both contribute to CNN learning
""")

print("Task 4 complete!")
print("\nAll plots saved to: plots/")
