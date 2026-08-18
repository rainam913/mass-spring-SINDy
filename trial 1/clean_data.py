import numpy as np
import pandas as pd

# -----------------------------
# 1. Load CSV
# -----------------------------
filename = "sindy iolab/trial 1/trail1_data_raw.csv"
df = pd.read_csv(filename)

# Clean column names
df.columns = df.columns.str.strip()

# -----------------------------
# 2. Select time + position
# -----------------------------
# CHANGE this if your position column is named differently
time = df["time"].values
position = df["cal[0]"].values   # usually cart position

# -----------------------------
# 3. Compute velocity
# -----------------------------
dt = np.gradient(time)
velocity = np.gradient(position) / dt

# -----------------------------
# 4. Detect stationary motion
# -----------------------------
velocity_threshold = 0.01  # m/s (tweak if needed)

moving_mask = np.abs(velocity) > velocity_threshold

# -----------------------------
# 5. Filter dataframe
# -----------------------------
df_clean = df[moving_mask].reset_index(drop=True)

# -----------------------------
# 6. Save cleaned data
# -----------------------------
output_file = "trial1_clean.csv"
df_clean.to_csv(output_file, index=False)

print(f"Removed stationary motion.")
print(f"Original samples: {len(df)}")
print(f"Remaining samples: {len(df_clean)}")
print(f"Saved as: {output_file}")
