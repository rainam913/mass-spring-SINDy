"""Exercise 31: regenerate the AIC/BIC table from the archived trial-3 fits.

Target: acceleration estimated from the recorded velocity of
mass_spring_final_tol005/running_my_data/cut trials/trial3_cut.csv
(N = 241) by the paper's Sec. 2 stencil -- centered differences in the
interior, first-order one-sided at the two ends. Ordinary least squares on
each of the four candidate libraries; RSS as printed in Exercise 31:

    {1, x, v}                       k=3  RSS = 97.23
    {1, x, v, x^2, v^2}             k=5  RSS = 96.27
    {1, x, v, x^3, v^3}             k=5  RSS = 95.87
    {1, x, v, x^2, v^2, x^3, v^3}   k=7  RSS = 95.30
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "mass_spring_final_tol005" / "running_my_data" / "cut trials" / "trial3_cut.csv")
df.columns = df.columns.str.strip()
t = df["time"].values
x = df["cal[0]"].values
v = df["cal[1]"].values
N = len(t)

a = np.empty_like(v)                       # Sec. 2 stencil on the velocity
a[0] = (v[1] - v[0]) / (t[1] - t[0])
a[-1] = (v[-1] - v[-2]) / (t[-1] - t[-2])
a[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])

LIBS = [
    ("{1, x, v} (linear)",             [np.ones(N), x, v]),
    ("{1, x, v, x^2, v^2}",            [np.ones(N), x, v, x**2, v**2]),
    ("{1, x, v, x^3, v^3} (custom)",   [np.ones(N), x, v, x**3, v**3]),
    ("{1, x, v, x^2, v^2, x^3, v^3}",  [np.ones(N), x, v, x**2, v**2, x**3, v**3]),
]
EXPECT = [97.23, 96.27, 95.87, 95.30]

print(f"N = {N}")
print(f"{'library':34} {'k':>2} {'RSS':>8} {'R^2':>8} {'AIC':>9} {'BIC':>9}")
TSS = np.sum((a - a.mean())**2)
for (name, cols), want in zip(LIBS, EXPECT):
    Theta = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(Theta, a, rcond=None)
    rss = float(np.sum((a - Theta @ beta)**2))
    k = Theta.shape[1]
    aic = 2 * k + N * np.log(rss / N)
    bic = k * np.log(N) + N * np.log(rss / N)
    print(f"{name:34} {k:>2} {rss:8.2f} {1 - rss/TSS:8.4f} {aic:9.2f} {bic:9.2f}")
    assert round(rss, 2) == want, f"{name}: RSS {rss:.2f} != printed {want}"
print("\nall four RSS values match Exercise 31 as printed")
