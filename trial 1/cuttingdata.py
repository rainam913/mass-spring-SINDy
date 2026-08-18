import numpy as np
import pandas as pd

# -----------------------------
# 1. Load CSV
# -----------------------------
filename = "trial1_clean.csv"
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
# 4. Detect motion AND filter by time
# -----------------------------
velocity_threshold = 0.01  # m/s
start_time = 4.7          # seconds

# Combined mask: moving AND time is >= 6.5
moving_mask = (np.abs(velocity) > velocity_threshold) & (time >= start_time)

# -----------------------------
# 5. Filter dataframe
# -----------------------------
df_clean = df[moving_mask].reset_index(drop=True)

# -----------------------------
# 6. Save cleaned data
# -----------------------------
output_file = "trial1_cut.csv"
df_clean.to_csv(output_file, index=False)

print(f"Removed stationary motion.")
print(f"Original samples: {len(df)}")
print(f"Remaining samples: {len(df_clean)}")
print(f"Saved as: {output_file}")
