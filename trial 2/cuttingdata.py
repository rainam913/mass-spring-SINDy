import numpy as np
import pandas as pd

# load csv
filename = "trial2_clean.csv"
df = pd.read_csv(filename)

# clean column names
df.columns = df.columns.str.strip()

# select time and position
time = df["time"].values
position = df["cal[0]"].values   # usually cart position

# compute velocity
dt = np.gradient(time)
velocity = np.gradient(position) / dt

# detect motion and filter by time
velocity_threshold = 0.01  # m/s
start_time = 4.7          # seconds

# Combined mask: moving AND time is >= 6.5
moving_mask = (np.abs(velocity) > velocity_threshold) & (time >= start_time)

# filter data frame
df_clean = df[moving_mask].reset_index(drop=True)

# save cleaned data
output_file = "trial2_cut.csv"
df_clean.to_csv(output_file, index=False)

print(f"Removed stationary motion.")
print(f"Original samples: {len(df)}")
print(f"Remaining samples: {len(df_clean)}")
print(f"Saved as: {output_file}")
