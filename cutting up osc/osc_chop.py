import numpy as np
import pandas as pd

# -----------------------------
# 1. Load CSV
# -----------------------------
filename = "trial3_clean.csv"
df = pd.read_csv(filename)

# Clean column names
df.columns = df.columns.str.strip()

# -----------------------------
# 2. Select time + position
# -----------------------------
time = df["time"].values
position = df["cal[0]"].values 

# -----------------------------
# 3. Compute velocity
# -----------------------------
dt = np.gradient(time)
velocity = np.gradient(position) / dt

# -----------------------------
# 4. Detect motion AND filter by time window
# -----------------------------
velocity_threshold = 0.01  # m/s
start_time = 5.5           # seconds
end_time = 7.0           # seconds (NEW: set your desired end point)

# Combined mask: moving AND within the [start, end] time window
moving_mask = (np.abs(velocity) > velocity_threshold) & \
              (time >= start_time) & \
              (time <= end_time)

# -----------------------------
# 5. Filter dataframe
# -----------------------------
df_clean = df[moving_mask].reset_index(drop=True)

# -----------------------------
# 6. Save cleaned data
# -----------------------------
output_file = "1_and_2.csv"
df_clean.to_csv(output_file, index=False)

print(f"Filtered data between {start_time}s and {end_time}s.")
print(f"Original samples: {len(df)}")
print(f"Remaining samples: {len(df_clean)}")
print(f"Saved as: {output_file}")