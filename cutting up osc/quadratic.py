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

# --- Improved Library Configuration ---
library_functions = [
    lambda x: x,        # Linear
    lambda x: x**2      # Quadratic
]
library_function_names = [
    lambda x: x, 
    lambda x: x + "^2"
]

custom_lib = ps.CustomLibrary(
    library_functions=library_functions, 
    function_names=library_function_names
)

# Initialize and fit model
model = ps.SINDy(
    differentiation_method=ps.SmoothedFiniteDifference(),
    feature_library=custom_lib,
    optimizer=ps.STLSQ(threshold=0.01) 
)

model.fit(X, t=t)
print("--- Linear + Quadratic SINDy Model Equations ---")
model.print()

# --- Custom R^2 Calculation Section ---
def compute_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    RSS = np.sum(residuals**2)
    TSS = np.sum((y_true - np.mean(y_true, axis=0))**2)
    return 1 - (RSS / TSS) if TSS > 0 else 0

X_dot_true = model.differentiation_method(X, t)
X_dot_pred = model.predict(X)

print("\n--- Model Performance (R^2 on Derivatives) ---")
for i in range(len(labels)):
    r2 = compute_metrics(X_dot_true[:, i], X_dot_pred[:, i])
    print(f"R^2 for d({labels[i]})/dt: {r2:.4f}")

# --- Simulation and Plotting ---
x0 = X[0]
try:
    # Use integrator_keywords to improve stability if needed
    X_sim = model.simulate(x0, t, integrator_kws={'rtol': 1e-6, 'atol': 1e-6})
    
    for i in range(2):
        plt.figure(figsize=(8, 4))
        plt.plot(t, X[:, i], "k", label="Original Data", alpha=0.7)
        plt.plot(t, X_sim[:, i], "r--", label="SINDy (Linear + Quad)")
        plt.xlabel("Time")
        plt.ylabel(labels[i])
        plt.title(f"Model Comparison: {labels[i]}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

except RuntimeError as e:
    print(f"\nSimulation failed: {e}")
    print("Hint: The system may still be unstable. Try adding a 'constant' term or increasing the STLSQ threshold.")