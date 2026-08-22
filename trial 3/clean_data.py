import numpy as np
import pandas as pd

# load csv
filename = "trial3_raw.csv"
df = pd.read_csv(filename)

# Clean column names
df.columns = df.columns.str.strip()

# select time and position
# CHANGE this if your position column is named differently
time = df["time"].values
position = df["cal[0]"].values   # usually cart position

# compute velocity
dt = np.gradient(time)
velocity = np.gradient(position) / dt

# detect stationary motion
velocity_threshold = 0.01  # m/s (tweak if needed)

moving_mask = np.abs(velocity) > velocity_threshold

# filter dataframe
df_clean = df[moving_mask].reset_index(drop=True)

# save cleaned data
output_file = "trial3_clean.csv"
df_clean.to_csv(output_file, index=False)

print(f"Removed stationary motion.")
print(f"Original samples: {len(df)}")
print(f"Remaining samples: {len(df_clean)}")
print(f"Saved as: {output_file}")
