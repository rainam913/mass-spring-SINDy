"""fit_tol005.py

SINDy/STLSQ identification of the IO-cart mass-spring system, run on BOTH
datasets in this folder (data_raw/, data_trimmed/), each PER TRIAL and
POOLED (all three trials fit jointly as multiple trajectories).

Model:       x' = f(x, v),  v' = g(x, v)
State:       raw:     x = cal[0] (m),      v = cal[1] (m/s)
             trimmed: x = position_m (m),  v = velocity_mps (m/s)
Derivatives: 2nd-order central finite differences of x and v (per trial)
Library:     full polynomial features of (x, v) through degree 3 (10 terms/eq)
Optimizer:   STLSQ, alpha = 0, normalize_columns = True (features scaled to
             unit norm inside the optimizer, so the threshold acts on effect
             size); threshold swept over 40 log-spaced values on [1e-3, 1e3]
Selection:   fewest-term model with derivative-fit R^2 within TOL = 0.005
             of the best model in the sweep

Reported per fit:
  - selected system and its second-order form
  - R^2 (derivative): pySINDy model.score -- pointwise fit of the library
    to the central-difference derivatives (both equations, uniform average)
  - R^2 (simulation): the selected system integrated from each trial's
    measured (x0, v0), simulated position vs measured position

Output:  results/report_tol005.txt

Run from the project folder:  python fit_tol005.py
"""
import sys, warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / 'pysindy_upstream' / 'pysindy').exists():      # prefer the vendored repo
    sys.path.insert(0, str(HERE / 'pysindy_upstream'))

import numpy as np
import pandas as pd
import pysindy as ps
from scipy.integrate import solve_ivp

warnings.filterwarnings('ignore')

# ----------------------------- configuration -----------------------------
TRIALS     = ['T1', 'T2', 'T3']
OUT        = HERE / 'results'
REPORT     = OUT / 'report_tol005.txt'
TOL        = 0.005
ALPHA      = 0.0
THRESHOLDS = np.logspace(-3, 3, 40)

POW    = [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3)]
NAMES  = ['1','x','v','x^2','x v','v^2','x^3','x^2 v','x v^2','v^3']
NAMES2 = ['1','x',"x'",'x^2',"x x'","x'^2",'x^3',"x^2 x'","x x'^2","x'^3"]

DATASETS = {
    'RAW':     dict(path=HERE / 'data_raw' / '{}_raw.csv',
                    tcol='time', xcol='cal[0]', vcol='cal[1]'),
    'TRIMMED': dict(path=HERE / 'data_trimmed' / '{}_trim.csv',
                    tcol='t_since_release', xcol='position_m', vcol='velocity_mps'),
}

# ----------------------------- helpers -----------------------------------
def load(spec, trial):
    d = pd.read_csv(str(spec['path']).format(trial), skipinitialspace=True)
    t = d[spec['tcol']].values
    return t - t[0], d[spec['xcol']].values, d[spec['vcol']].values

def central_diff(y, t):
    """2nd-order central differences (one-sided 2nd-order at the ends)."""
    return ps.FiniteDifference(order=2)._differentiate(y.reshape(-1, 1), t).ravel()

def sweep(X_list, Xd_list, T_list):
    rows = []
    for th in THRESHOLDS:
        m = ps.SINDy(optimizer=ps.STLSQ(threshold=th, alpha=ALPHA, max_iter=50,
                                        normalize_columns=True),
                     feature_library=ps.PolynomialLibrary(degree=3))
        m.fit(X_list, t=T_list, x_dot=Xd_list)
        rows.append((th, int(np.count_nonzero(m.coefficients())),
                     m.score(X_list, t=T_list, x_dot=Xd_list), m))
    return rows

def select(rows):
    best = max(r[2] for r in rows)
    ok = [r for r in rows if r[2] >= best - TOL and r[1] > 0]
    ok.sort(key=lambda r: (r[1], -r[2]))          # fewest terms, then best R^2
    return ok[0], best

def eq_str(c, names, prec=4):
    js = np.nonzero(c)[0]
    if len(js) == 0:
        return '0'
    return ' '.join(f'{c[j]:+.{prec}g} {names[j]}'.replace(' 1', '') for j in js)

def second_order_lines(C):
    lines = [f"    x'' = {eq_str(C[1], NAMES2)}"]
    c = C[1]
    if all(c[j] == 0 for j in range(3, 10)) and c[1] < 0:   # linear in (1, x, v)
        k, b, F = -c[1], -c[2], c[0]
        w, z = np.sqrt(k), b / (2 * np.sqrt(k))
        lines.append(f"    standard form: x'' + {b:.4g} x' + {k:.4g} x = {F:.4g}")
        lines.append(f"    omega_n = {w:.4g} rad/s  (f = {w/2/np.pi:.4g} Hz, "
                     f"T = {2*np.pi/w:.4g} s), zeta = {z:.4g}, x* = {-c[0]/c[1]:.4g} m")
    else:
        lines.append("    (nonlinear terms present - no single (omega_n, zeta) summary)")
    return lines

def simulate_r2(C, t, x, v):
    def rhs(t_, s):
        if abs(s[0]) > 10 or abs(s[1]) > 100:               # blow-up guard
            return [0.0, 0.0]
        theta = np.array([s[0]**p * s[1]**q for p, q in POW])
        return C @ theta
    sol = solve_ivp(rhs, (t[0], t[-1]), [x[0], v[0]], t_eval=t,
                    rtol=1e-8, atol=1e-10)
    if not (sol.success and sol.y.shape[1] == len(t)) or np.abs(sol.y[0]).max() > 5:
        return None
    xs = sol.y[0]
    return 1 - np.sum((x - xs)**2) / np.sum((x - x.mean())**2)

def sweep_table(rows, chosen):
    out = [f'{"thresh":>10} {"terms":>6} {"R2 (deriv)":>11}']
    seen = set()
    for row in rows:
        th, nnz, r2 = row[0], row[1], row[2]
        key = (nnz, round(r2, 6))
        mark = '  <-- selected' if row is chosen else ''
        if key in seen and not mark:
            continue
        seen.add(key)
        out.append(f'{th:10.3f} {nnz:6d} {r2:11.6f}{mark}')
    return out

def fit_and_report(tag, segments):
    """segments: list of (t, x, v); pooled if len > 1."""
    X_list  = [np.column_stack([x, v]) for (t, x, v) in segments]
    Xd_list = [np.column_stack([central_diff(x, t), central_diff(v, t)])
               for (t, x, v) in segments]
    T_list  = [t for (t, x, v) in segments]

    rows = sweep(X_list, Xd_list, T_list)
    chosen, best = select(rows)
    th, nnz, r2, m = chosen
    C = m.coefficients()

    lines = ['', f'--- {tag} ' + '-' * max(2, 60 - len(tag)), '']
    lines += sweep_table(rows, chosen)
    lines += ['',
              f'Selected: threshold = {th:g}, {nnz} nonzero terms',
              f'R^2 (derivative) = {r2:.6f}   (best in sweep {best:.6f})',
              '',
              'Identified system:',
              f"    x' = {eq_str(C[0], NAMES)}",
              f"    v' = {eq_str(C[1], NAMES)}",
              '',
              'Second-order form:']
    lines += second_order_lines(C)
    lines += ['', 'R^2 (simulation), position trajectory from each measured (x0, v0):']
    for trial, (t, x, v) in zip(TRIALS if len(segments) > 1 else [tag.split()[-1]],
                                segments):
        r2s = simulate_r2(C, t, x, v)
        lines.append(f'    {trial}: ' + ('DIVERGES / fails' if r2s is None
                                         else f'{r2s:.4f}'))
    return lines

# ----------------------------- main --------------------------------------
OUT.mkdir(exist_ok=True)
report = [
    'SINDy/STLSQ identification, full cubic library  (fit_tol005.py)',
    f'pysindy {ps.__version__} | polynomial degree 3 | central-difference '
    f'derivatives | STLSQ: alpha = {ALPHA}, normalize_columns = True | '
    f'TOL = {TOL}',
    'R^2 (derivative): pointwise fit to the differenced derivatives '
    '(both equations, uniform average).',
    'R^2 (simulation): selected model integrated from measured (x0, v0), '
    'simulated vs measured position.',
    '=' * 78,
]

for ds_name, spec in DATASETS.items():
    report += ['', '#' * 78, f'##  DATASET: {ds_name}', '#' * 78]
    segments = {trial: load(spec, trial) for trial in TRIALS}
    for trial in TRIALS:
        report += fit_and_report(f'{ds_name} {trial}', [segments[trial]])
    report += fit_and_report(f'{ds_name} POOLED (T1+T2+T3)',
                             [segments[trial] for trial in TRIALS])

REPORT.write_text('\n'.join(report) + '\n')
print(f'wrote {REPORT}')
