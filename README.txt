# mass_spring_SINDy

Using **SINDy** (Sparse Identification of Nonlinear Dynamics) to discover the governing equations of motion for a mass-spring oscillator system from experimental iOLab cart data.

## Overview

This project takes raw position data collected from an iOLab cart on a spring track and uses [PySINDy](https://pysindy.readthedocs.io/) to fit a sparse, interpretable model of the system's dynamics — i.e. it tries to recover an equation of the form `dx/dt = f(x, y)` directly from data, rather than assuming a fixed physical model up front.

The workflow across the repo is generally:
1. **Clean** raw trial data (strip whitespace from columns, trim stationary/non-moving segments).
2. **Segment** cleaned data into windows of interest (e.g. a single oscillation, or a transition between regimes).
3. **Fit** a SINDy model to the segmented data using different candidate feature libraries (linear, quadratic, cubic, fifth-order).
4. **Evaluate** the fit (R² on derivatives, simulated trajectory vs. actual, percent error) and plot results.

## Repository Structure

```
├── trial 1/ trial 2/ trial 3/   # Raw experimental trials
│   ├── *_raw.csv                # Raw iOLab sensor output
│   ├── clean_data.py            # Strips stationary segments from raw data
│   ├── *_clean.csv / *_cut.csv  # Cleaned / windowed data
│   ├── cuttingdata.py           # Trims data to a specific time window
│   ├── iolab_reg.py             # Fits a SINDy model (PolynomialLibrary + STLSQ)
│   └── iolab_damping.py         # Damping-related analysis
│
├── cutting up osc/               # Segmenting a trial into individual oscillation windows
│   ├── osc_chop.py               # Splits trial data into time-windowed segments
│   ├── 0_to_1.csv, 1_to_2.csv, 1_and_2.csv   # Individual segment outputs
│   └── linear.py, quadratic.py, cubics.py, fifth_order.py  # SINDy fits per segment, by library order
│
└── stlsq/                        # Refined SINDy fitting approach
    ├── oscillator_library.py     # Shared helpers: symmetric feature library + custom optimizer
    ├── linear.py, quadratic.py, cubics.py, fifth_order.py  # Fits using the shared library/optimizer
    └── graphs.py                 # Plots raw time-series data
```

### `stlsq/oscillator_library.py`

The core of the more refined fitting approach. It provides:

- **`build_symmetric_library(powers, ...)`** — builds a `CustomLibrary` that applies the same set of power terms (e.g. `x`, `x^3`) symmetrically to both state channels (`cal[0]` = x, `cal[1]` = y), since both are spatial coordinates of the same physical type rather than a position/velocity pair.
- **`PostSelectionSTLSQ`** — a variant of PySINDy's `STLSQ` optimizer that performs sparse term selection as usual, then refits the surviving terms with plain ordinary least squares (no ridge bias). This avoids depending on the `unbias=True` argument, which isn't available in older PySINDy versions.
- **`make_post_selection_optimizer(threshold)`** — convenience wrapper combining column normalization (so no term is favored purely due to numeric scale) with the post-selection refit above.
- **`fit_sindy_model(model, X, t, feature_names)`** — fits a model in a way that works across PySINDy versions that disagree on whether `feature_names` is a `fit()` argument or a constructor argument.

## Requirements

```
numpy
pandas
matplotlib
pysindy
```

Install with:
```bash
pip install numpy pandas matplotlib pysindy
```

## Usage

Each trial/segment script expects to be run from within its own folder, since data is loaded via relative paths (e.g. `pd.read_csv("0_to_1.csv")`). For example:

```bash
cd stlsq
python linear.py
```

This will print the discovered model equations, R² values for the fit, and (depending on the script) show plots comparing the simulated trajectory to the actual data.

> **Note:** `stlsq/graphs.py` currently references an absolute local path (`/Users/rainamajumder/...`) — update this to a relative path before running it elsewhere.

## Notes

- Scripts are organized experimentally (trial-by-trial, segment-by-segment) rather than as a single reusable pipeline — expect some duplication between folders as different feature libraries and cleaning approaches were tried.
- `include_xy_cross_term` in `oscillator_library.py` is available if you want to test for explicit `x*y` coupling terms beyond what a linear combination of `{x, y, x^3, y^3, ...}` can already represent.
