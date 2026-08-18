import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt

from oscillator_library import build_symmetric_library, make_post_selection_optimizer, fit_sindy_model

# Load and clean data
df = pd.read_csv("0_to_1.csv")
df.columns = df.columns.str.strip()

t = df["time"].values
labels = ["cal[0]", "cal[1]"]  # cal[0] = x axis, cal[1] = y axis
X = df[labels].values

# Library: {x, y, x^3, y^3} -- odd terms only, applied symmetrically to
# both axes since they're the same physical type of quantity.
custom_lib = build_symmetric_library(powers=[1, 3], include_bias=False)

# Post-selection least squares (see oscillator_library.py for what these two flags do)
optimizer = make_post_selection_optimizer(threshold=0.05)

model = ps.SINDy(
    differentiation_method=ps.SmoothedFiniteDifference(),
    feature_library=custom_lib,
    optimizer=optimizer,
)

fit_sindy_model(model, X, t, feature_names=["x", "y"])
print("--- SINDy Model Equations ---")
model.print()

# --- Custom R^2 Calculation Section (Derivative-Based) ---


def compute_metrics(y_true, y_pred):
    """
    Computes R^2 based on numerical derivatives (y_true)
    vs analytically predicted derivatives (y_pred).
    """
    residuals = y_true - y_pred
    RSS = np.sum(residuals**2)
    TSS = np.sum((y_true - np.mean(y_true)) ** 2)
    R2 = 1 - (RSS / TSS) if TSS > 0 else 0
    return R2


# 1. Get numerically estimated derivatives from data (y_true)
X_dot_true = model.differentiation_method(X, t)

# 2. Get analytically predicted derivatives from the discovered ODE (y_pred)
X_dot_pred = model.predict(X)

print("\n--- Model Performance (R^2 on Derivatives) ---")
derivative_r2_scores = []
for i in range(len(labels)):
    r2 = compute_metrics(X_dot_true[:, i], X_dot_pred[:, i])
    derivative_r2_scores.append(r2)
    print(f"R^2 for d({labels[i]})/dt: {r2:.4f}")

# --- Simulation and Plotting ---

x0 = X[0]
try:
    X_sim = model.simulate(x0, t)

    for i in range(2):
        plt.figure(figsize=(8, 4))
        plt.plot(t, X[:, i], "k", label="Original Data", alpha=0.7)
        plt.plot(t, X_sim[:, i], "--b", label=f"SINDy Prediction (Deriv R²={derivative_r2_scores[i]:.3f})")
        plt.xlabel("Time")
        plt.ylabel(labels[i])
        plt.title(f"Cubic Fit Comparison: {labels[i]}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

except RuntimeError as e:
    print(f"\nSimulation failed: {e}")
