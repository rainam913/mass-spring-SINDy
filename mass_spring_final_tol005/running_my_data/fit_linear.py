"""fit_linear.py  (cut-trials version)

Restricted LINEAR-library identification on the three cut trials
(cut trials/trial{1,2,3}_cut.csv), per trial and pooled.

    x' = a0 + a1 x + a2 v
    v' = c0 + c1 x + c2 v      (library: 1, x, v)

Coefficients estimated by ordinary least squares (STLSQ with
threshold = 0, alpha = 0 -- no sparsification left to do).

State:       x = cal[0], v = cal[1]
Derivatives: 2nd-order central finite differences of x and v (per trial)

Reported per fit: the system, its second-order standard form
(omega_n, zeta, f, period, equilibrium x*), R^2 (derivative), and
R^2 (simulation) from each trial's measured (x0, v0).

Output:  results/report_linear.txt

Run from the project folder:  python fit_linear.py
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
TRIALS = ['trial1', 'trial2', 'trial3']
DATA   = HERE / 'cut trials' / '{}_cut.csv'
TCOL, XCOL, VCOL = 'time', 'cal[0]', 'cal[1]'
OUT    = HERE / 'results'
REPORT = OUT / 'report_linear.txt'

NAMES  = ['1', 'x', 'v']
NAMES2 = ['1', 'x', "x'"]

# ----------------------------- helpers -----------------------------------
def load(trial):
    d = pd.read_csv(str(DATA).format(trial), skipinitialspace=True)
    t = d[TCOL].values
    return t - t[0], d[XCOL].values, d[VCOL].values

def central_diff(y, t):
    """2nd-order central differences (one-sided 2nd-order at the ends)."""
    return ps.FiniteDifference(order=2)._differentiate(y.reshape(-1, 1), t).ravel()

def eq_str(c, names, prec=4):
    js = np.nonzero(c)[0]
    if len(js) == 0:
        return '0'
    return ' '.join(f'{c[j]:+.{prec}g} {names[j]}'.replace(' 1', '') for j in js)

def simulate_r2(C, t, x, v):
    def rhs(t_, s):
        return C @ np.array([1.0, s[0], s[1]])
    sol = solve_ivp(rhs, (t[0], t[-1]), [x[0], v[0]], t_eval=t,
                    rtol=1e-8, atol=1e-10)
    if not (sol.success and sol.y.shape[1] == len(t)):
        return None
    xs = sol.y[0]
    return 1 - np.sum((x - xs)**2) / np.sum((x - x.mean())**2)

def fit_and_report(tag, segments):
    """segments: list of (t, x, v); pooled if len > 1."""
    X_list  = [np.column_stack([x, v]) for (t, x, v) in segments]
    Xd_list = [np.column_stack([central_diff(x, t), central_diff(v, t)])
               for (t, x, v) in segments]
    T_list  = [t for (t, x, v) in segments]

    m = ps.SINDy(optimizer=ps.STLSQ(threshold=0.0, alpha=0.0),
                 feature_library=ps.PolynomialLibrary(degree=1))
    m.fit(X_list, t=T_list, x_dot=Xd_list)
    r2d = m.score(X_list, t=T_list, x_dot=Xd_list)
    C = m.coefficients()

    c = C[1]                       # v' = c0 + c1 x + c2 v
    k, b, F = -c[1], -c[2], c[0]
    w  = np.sqrt(k)
    z  = b / (2 * w)
    xe = F / k

    lines = ['', f'--- {tag} ' + '-' * max(2, 60 - len(tag)), '',
             'Identified system:',
             f"    x' = {eq_str(C[0], NAMES)}",
             f"    v' = {eq_str(C[1], NAMES)}",
             '',
             'Second-order form:',
             f"    x'' + {b:.4g} x' + {k:.4g} x = {F:.4g}",
             f'    omega_n = {w:.4g} rad/s   f = {w/2/np.pi:.4g} Hz   '
             f'period = {2*np.pi/w:.4g} s',
             f'    zeta = {z:.4g}   equilibrium x* = {xe:.4g} m',
             '',
             f'R^2 (derivative) = {r2d:.4f}',
             'R^2 (simulation), position trajectory from each measured (x0, v0):']
    for trial, (t, x, v) in zip(TRIALS if len(segments) > 1 else [tag.split()[-1]],
                                segments):
        r2s = simulate_r2(C, t, x, v)
        lines.append(f'    {trial}: ' + ('simulation failed' if r2s is None
                                         else f'{r2s:.4f}'))
    return lines, (k, b, F, w, z, xe)

# ----------------------------- main --------------------------------------
OUT.mkdir(exist_ok=True)
report = [
    'Restricted linear-library identification, cut trials  (fit_linear.py)',
    f'pysindy {ps.__version__} | polynomial degree 1 (library: 1, x, v) | '
    'central-difference derivatives | ordinary least squares '
    '(STLSQ threshold = 0, alpha = 0)',
    'R^2 (derivative): pointwise fit to the differenced derivatives '
    '(both equations, uniform average).',
    'R^2 (simulation): identified model integrated from measured (x0, v0), '
    'simulated vs measured position.',
    '=' * 78,
]

segments = {trial: load(trial) for trial in TRIALS}
params = {}
for trial in TRIALS:
    lines, p = fit_and_report(f'CUT {trial}', [segments[trial]])
    report += lines
    params[trial] = p
lines, p = fit_and_report('CUT POOLED (trial1+trial2+trial3)',
                          [segments[trial] for trial in TRIALS])
report += lines
params['POOLED'] = p

report += ['', '--- CROSS-TRIAL SUMMARY ' + '-' * 38, '',
           f'{"fit":>8} {"k = -c1":>9} {"b = -c2":>9} {"F = c0":>9} '
           f'{"omega_n":>9} {"zeta":>8} {"x* (m)":>10}']
for tag, (k, b, F, w, z, xe) in params.items():
    report.append(f'{tag:>8} {k:9.2f} {b:9.3f} {F:9.3f} {w:9.3f} {z:8.4f} {xe:10.5f}')
ks = [params[tr][0] for tr in TRIALS]
bs = [params[tr][1] for tr in TRIALS]
report += ['',
           f'per-trial spread: k = {np.mean(ks):.2f} +/- {np.std(ks):.2f}  '
           f'({100*np.std(ks)/np.mean(ks):.1f}%),  '
           f'b = {np.mean(bs):.3f} +/- {np.std(bs):.3f}  '
           f'({100*np.std(bs)/np.mean(bs):.1f}%)']

REPORT.write_text('\n'.join(report) + '\n')
print(f'wrote {REPORT}')