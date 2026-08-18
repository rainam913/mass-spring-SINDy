import numpy as np
import pysindy as ps


def _make_power_fn(power):
    """Return a genuine single-argument function a -> a**power."""
    if power == 1:
        return lambda a: a
    return lambda a: a ** power


def _make_power_name_fn(power):
    """Matching name function: e.g. 'x' -> 'x', or 'x' -> 'x^3'."""
    if power == 1:
        return lambda a: a
    return lambda a: f"{a}^{power}"


def build_symmetric_library(powers, include_bias=False, include_xy_cross_term=False):

    fns = [_make_power_fn(p) for p in powers]
    names = [_make_power_name_fn(p) for p in powers]

    if include_xy_cross_term:
        fns = fns + [lambda a, b: a * b]
        names = names + [lambda a, b: f"{a}*{b}"]

    if include_bias:
        fns = [lambda a: np.ones_like(a)] + fns
        names = [lambda a: "1"] + names

    return ps.CustomLibrary(library_functions=fns, function_names=names)


class PostSelectionSTLSQ(ps.STLSQ):

    def _reduce(self, x, y):
        super()._reduce(x, y)  # normal STLSQ selection (may be ridge-biased)
        n_targets, n_features = self.coef_.shape
        refit_coef = np.zeros_like(self.coef_)
        for i in range(n_targets):
            support = np.abs(self.coef_[i]) > 1e-14
            if np.any(support):
                coef_i, *_ = np.linalg.lstsq(x[:, support], y[:, i], rcond=None)
                refit_coef[i, support] = coef_i
        self.coef_ = refit_coef


def make_post_selection_optimizer(threshold):
    return PostSelectionSTLSQ(threshold=threshold, normalize_columns=True)


def fit_sindy_model(model, X, t, feature_names):

    try:
        model.fit(X, t=t, feature_names=feature_names)
    except TypeError:
        model.feature_names = feature_names
        model.fit(X, t=t)
    return model
