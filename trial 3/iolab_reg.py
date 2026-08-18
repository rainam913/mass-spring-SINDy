import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("trial3_clean.csv")
df.columns = df.columns.str.strip()

t = df["time"].values
X = df[["cal[0]", "cal[1]"]].values

# Build SINDy model
model = ps.SINDy(
    differentiation_method=ps.SmoothedFiniteDifference(),
    feature_library=ps.PolynomialLibrary(
        degree=3,
        include_interaction=False,  # removes cross terms
        include_bias=True
    ),
    optimizer=ps.STLSQ(threshold=0.05)
)

# Fit model
model.fit(X, t=t)
model.print()

# Simulate model
x0 = X[0]
t_sim = np.linspace(t[0], t[-1], len(t))
X_sim = model.simulate(x0, t_sim)

# Compute Absolute Percent Error
eps = 1e-8
APE = np.abs(X - X_sim) / (np.abs(X) + eps) * 100

APE_exceeds_3 = APE > 3.0
labels = ["cal[0]", "cal[1]"]

# Indices where APE > 3% for cal[1]
np.where(APE_exceeds_3[:, 1])

# Plot APE
for i in range(2):
    plt.figure()
    plt.plot(t, APE[:, i])
    plt.xlabel("time")
    plt.ylabel("Absolute Percent Error (%)")
    plt.title(f"APE for {labels[i]}")
    plt.show()

# Plot data vs SINDy model
for i in range(2):
    plt.figure()
    plt.plot(t, X[:, i], "k", label="data")
    plt.plot(t_sim, X_sim[:, i], "--r", label="SINDy")
    plt.xlabel("time")
    plt.ylabel(labels[i])
    plt.legend()
    plt.show()

# Find time steps where APE > 3%
exceed_list = []

for i in range(2):
    idx = np.where(APE[:, i] > 3.0)[0]
    for j in idx:
        exceed_list.append({
            "time": t[j],
            "state": labels[i],
            "APE (%)": APE[j, i]
        })

exceed_df = pd.DataFrame(exceed_list)
print(exceed_df)