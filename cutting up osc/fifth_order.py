import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv("1_and_2.csv")
df.columns = df.columns.str.strip()

t = df["time"].values
labels = ["cal[0]", "cal[1]"]
X = df[labels].values

# Updated library: includes constant, linear, cubic, and fifth-order terms
library_functions = [
    lambda x: np.ones_like(x), # constant
    lambda x: x,               # linear (x)
    lambda x: x**3,            # cubic (x^3)
    lambda x: x**5             # fifth-order (x^5)
]

library_function_names = [
    lambda x: "1",
    lambda x: x,
    lambda x: x + "^3",
    lambda x: x + "^5"
]

custom_lib = ps.CustomLibrary(
    library_functions=library_functions, 
    function_names=library_function_names
)

# Initialize and fit model
model = ps.SINDy(
    differentiation_method=ps.SmoothedFiniteDifference(),
    feature_library=custom_lib,
    optimizer=ps.STLSQ(threshold=0.05)
)

model.fit(X, t=t)
print("--- SINDy Model Equations (up to 5th Order) ---")
model.print()

# --- Custom R^2 Calculation Section (Derivative-Based) ---

def compute_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    RSS = np.sum(residuals**2)
    TSS = np.sum((y_true - np.mean(y_true))**2)
    R2 = 1 - (RSS / TSS) if TSS > 0 else 0
    return R2

# 1. Get numerically estimated derivatives from data
X_dot_true = model.differentiation_method(X, t)

# 2. Get analytically predicted derivatives from the model
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
        plt.plot(t, X_sim[:, i], "--m", label=f"5th Order SINDy (R²={derivative_r2_scores[i]:.3f})")
        plt.xlabel("Time")
        plt.ylabel(labels[i])
        plt.title(f"Fifth-Order Fit Comparison: {labels[i]}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        
except RuntimeError as e:
    print(f"\nSimulation failed: {e}")
    print("Hint: 5th-order terms can make the ODE very 'stiff' or unstable during integration.")