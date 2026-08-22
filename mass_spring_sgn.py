"""Exercise 25: finish the dry-friction investigation in code.

Adds a smoothed signum feature, tanh(v / eps), to the paper's custom
mass-spring library {1, x, v, x^3, v^3} and reruns the identification on
all three trials (the cut records of mass_spring_final_tol005/running_my_data).
If the persistent v^3 term of Sec. 6 is an odd-polynomial imitation of
Coulomb friction, it should shrink or vanish once the library can express
sgn(v) directly, and the recovered friction coefficient should be
comparable across trials.

Usage: python mass_spring_sgn.py [eps]      (default eps = 0.05 m/s)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps

EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
HERE = Path(__file__).parent
CUT = HERE / "mass_spring_final_tol005" / "running_my_data" / "cut trials"

def build_library(eps):
    # NOTE: CustomLibrary applies every function to EACH state variable, so a
    # constant lambda here would emit one constant column per state, i.e. two
    # identical, perfectly collinear columns. The constant is supplied once,
    # separately, via a degree-0 polynomial library.
    fns = [
        lambda x: x,
        lambda x: x**3,
        lambda x: np.tanh(x / eps),
    ]
    names = [
        lambda x: x,
        lambda x: x + "^3",
        lambda x: f"tanh({x}/{eps})",
    ]
    custom = ps.CustomLibrary(library_functions=fns, function_names=names)
    const = ps.PolynomialLibrary(degree=0, include_bias=True)
    return ps.GeneralizedLibrary([const, custom])

for trial in (1, 2, 3):
    df = pd.read_csv(CUT / f"trial{trial}_cut.csv")
    df.columns = df.columns.str.strip()
    t = df["time"].values
    X = df[["cal[0]", "cal[1]"]].values          # position, velocity
    model = ps.SINDy(
        differentiation_method=ps.SmoothedFiniteDifference(),
        feature_library=build_library(EPS),
        optimizer=ps.STLSQ(threshold=0.05, alpha=0, normalize_columns=True),
    )
    model.fit(X, t=t)
    print(f"--- Trial {trial} (eps = {EPS}) ---")
    model.print()
    print()
