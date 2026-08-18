import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt

df = pd.read_csv("trial1_data_clean.csv")
df.columns = df.columns.str.strip()

t = df["time"].values
X = df[["cal[0]", "cal[1]"]].values

# custom library to only include damping terms
# x[0] is cal[0], x[1] is cal[1]
library_functions = [
    lambda x: np.ones_like(x), # constant
    lambda x: x,               # y and ydot
    lambda x: x**3             # y^3 and ydot^3
]

library_function_names = [
    lambda x: "1",
    lambda x: x,
    lambda x: x + "^3"
]

custom_lib = ps.CustomLibrary(
    library_functions=library_functions, 
    function_names=library_function_names
)
# -------------------------------------

model = ps.SINDy(
    differentiation_method=ps.SmoothedFiniteDifference(),
    feature_library=custom_lib,
    optimizer=ps.STLSQ(threshold=0.05)
)

model.fit(X, t=t)
model.print()

# Simulation and Error Calculation
x0 = X[0]
t_sim = np.linspace(t[0], t[-1], len(t))
X_sim = model.simulate(x0, t_sim)

eps = 1e-8
APE = np.abs(X - X_sim) / (np.abs(X) + eps) * 100

# Results output
labels = ["cal[0]", "cal[1]"]
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

# Plotting
for i in range(2):
    plt.figure()
    plt.plot(t, X[:, i], "k", label="data")
    plt.plot(t_sim, X_sim[:, i], "--r", label="SINDy")
    plt.xlabel("time")
    plt.ylabel(labels[i])
    plt.title(f"Data vs Model: {labels[i]}")
    plt.legend()
    plt.show()