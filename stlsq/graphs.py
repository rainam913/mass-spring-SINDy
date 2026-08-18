import pandas as pd
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv("/Users/rainamajumder/Desktop/sindy things/sindy iolab/trial 3/trial3_raw.csv")
df.columns = df.columns.str.strip()

t = df["time"].values
labels = ["cal[0]", "cal[1]"]

# Create a figure with two side-by-side subplots (or change to (2, 1) to stack them vertically)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for i, label in enumerate(labels):
    axes[i].plot(t, df[label].values, color="black", alpha=0.8, label="Data")
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel(label)
    axes[i].set_title(f"Time Series: {label}")
    axes[i].grid(True, linestyle="--", alpha=0.6)
    axes[i].legend()

plt.tight_layout()
plt.show()