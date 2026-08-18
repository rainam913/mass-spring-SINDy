import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt


plt.rcParams.update({
    "figure.dpi": 180,        # sharper display in notebook
    "savefig.dpi": 400       # high-res if saved
})

# load data
df = pd.read_csv("trial3_cut.csv")
df.columns = df.columns.str.strip()

t = df["time"].values
X = df[["cal[0]", "cal[1]"]].values

# custom library
library_functions = [
    lambda x: np.ones_like(x),   # constant
    lambda x: x,                # linear
    lambda x: x**3              # cubic
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

# train model
model = ps.SINDy(
    differentiation_method=ps.SmoothedFiniteDifference(),
    feature_library=custom_lib,
    optimizer=ps.STLSQ(threshold=0.05)
)

model.fit(X, t=t)
model.print()

# simulate
x0 = X[0]
t_sim = np.linspace(t[0], t[-1], len(t))
X_sim = model.simulate(x0, t_sim)

# error calculations
eps = 1e-8
APE = np.abs(X - X_sim) / (np.abs(X) + eps) * 100

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

# plots
for i in range(2):
    plt.figure(figsize=(10, 6))
    
    plt.plot(t, X[:, i], color="black", label="Experimental Data")
    plt.plot(t_sim, X_sim[:, i], "--", color="red", label="SINDy Model")
    
    plt.xlabel("Time")
    plt.ylabel(labels[i])
    plt.title(f"Experimental vs SINDy Prediction: {labels[i]}")
    plt.legend(frameon=False)
    plt.tight_layout()
    
    # optional save
    plt.savefig(f"{labels[i]}_high_quality.png", bbox_inches="tight")
    
    plt.show()