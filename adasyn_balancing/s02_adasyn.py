"""Step 2 - ADASYN (Adaptive Synthetic Sampling), applied to the TRAINING set only.

How ADASYN works (He et al., 2008): for each minority sample it looks at its k nearest neighbours in the whole
training set. The share of majority neighbours is its "difficulty". Samples that are harder to learn receive
proportionally more synthetic points, created by linear interpolation towards a random minority neighbour.

Implementation choices:
* `imblearn.over_sampling.ADASYN`, one call PER minority class (sampling_strategy={class: n}). If ADASYN cannot
  produce samples for one class (e.g. none of its points has a majority neighbour), only that class is skipped
  and the reason is reported, instead of the whole run failing.
* Distances: only the NUMERIC columns (continuous + integer) are standardised before ADASYN, so the year or
  businesses-per-1000 cannot dominate the k-NN distances. Binary and one-hot columns stay 0/1: standardising them
  turned a rare level into a huge value (e.g. a category with 2 training rows -> 308) that dominated every
  distance. Left as 0/1, a category mismatch adds the same distance whatever the category's frequency.
  The scaling is affine and applied column-wise, so interpolating in the scaled space and inverting gives exactly
  the interpolation in the original space (checked in the tests).
* Small classes: k = min(n_neighbors, n_class - 1). A class with fewer than 2 rows cannot be interpolated and is
  left as it is.
* Mixed feature types: ADASYN interpolates, so one-hot and binary columns become fractions. Synthetic rows are
  repaired so the feature structure stays valid:
      one-hot group -> 1 for the largest value, 0 for the others
      binary        -> rounded to 0/1
      integer       -> rounded, clipped to the training range
      continuous    -> clipped to the training range
      registration  -> year and month rounded JOINTLY as one date (see below)
  and derived columns are recomputed: owners_total = Total M + Total F, has_female_owner = Total F > 0.
* Age / registration-date consistency. In Part I, age_years = (end - registration date) / 365.25 with
  end <= reference date (05.10.2026), so every real business satisfies age <= time since registration.
  This constraint is linear, so a linear interpolation between two real businesses satisfies it too. The
  violations came from the repair: year and month were rounded SEPARATELY (e.g. 2005.5 / 6.5 -> 2006 / June,
  almost a year after the interpolated date 2005-12). The repair now (1) converts year + month into one month
  index, rounds it once and splits it again, which moves the date by at most half a month, and then (2) caps the
  age at the time between that registration date and the reference date.
* Original training rows are returned unchanged; synthetic rows are appended and flagged.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler

from common.columns import C
from common.config import RANDOM_STATE, REFERENCE_DATE
from common.utils import show, step

N_NEIGHBORS = 5


class NumericScaler:
    """Standardise the numeric columns only; binary and one-hot columns pass through as 0/1."""

    def __init__(self, spec):
        self.numeric = spec.continuous + spec.integer

    def fit(self, X: pd.DataFrame):
        self.columns = list(X.columns)
        self.idx = [self.columns.index(c) for c in self.numeric]
        self.scaler = StandardScaler().fit(X[self.numeric])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        A = X[self.columns].to_numpy(dtype=float).copy()
        A[:, self.idx] = self.scaler.transform(X[self.numeric])
        return A

    def inverse_transform(self, A: np.ndarray) -> np.ndarray:
        A = np.array(A, dtype=float, copy=True)
        A[:, self.idx] = self.scaler.inverse_transform(A[:, self.idx])
        return A


def max_age_years(year: pd.Series, month: pd.Series) -> pd.Series:
    """Longest possible age (years) of a business registered in year/month: until the reference date."""
    reg = pd.to_datetime(dict(year=year.astype(int), month=month.astype(int), day=1))
    return (REFERENCE_DATE - reg).dt.days / 365.25


def _repair_registration_date(synth: pd.DataFrame, X_train: pd.DataFrame) -> pd.DataFrame:
    """Round year + month jointly (one month index), keep it in the training range, then cap the age."""
    y, m = C.YEAR, "registration_month"
    if not {y, m} <= set(synth.columns):
        return synth
    month_index = (synth[y] * 12 + (synth[m] - 1)).round()
    train_index = X_train[y] * 12 + (X_train[m] - 1)
    month_index = month_index.clip(train_index.min(), train_index.max())
    synth[y] = (month_index // 12).astype(float)
    synth[m] = (month_index % 12 + 1).astype(float)
    if "age_years" in synth.columns:
        synth["age_years"] = np.minimum(synth["age_years"], max_age_years(synth[y], synth[m])).clip(lower=0)
    return synth


def _repair(synth: pd.DataFrame, X_train: pd.DataFrame, spec) -> pd.DataFrame:
    """Turn interpolated values back into valid feature values (see module docstring)."""
    lo, hi = X_train.min(), X_train.max()
    synth = _repair_registration_date(synth, X_train)          # before the generic integer rounding
    for cols in spec.onehot_groups.values():
        block = synth[cols].to_numpy()
        onehot = np.zeros_like(block)
        onehot[np.arange(len(block)), block.argmax(axis=1)] = 1.0
        synth[cols] = onehot
    for c in spec.binary:
        synth[c] = synth[c].round().clip(0, 1)
    for c in spec.integer:                                      # year / month are already whole numbers
        synth[c] = synth[c].round().clip(lo[c], hi[c])
    for c in spec.continuous:
        synth[c] = synth[c].clip(lo[c], hi[c])
    if {C.OWNERS_M, C.OWNERS_F, "owners_total"} <= set(synth.columns):
        synth["owners_total"] = synth[C.OWNERS_M] + synth[C.OWNERS_F]
    if {"has_female_owner", C.OWNERS_F} <= set(synth.columns):
        synth["has_female_owner"] = (synth[C.OWNERS_F] > 0).astype(float)
    return synth


def resample(X_train: pd.DataFrame, y_train: pd.Series, spec, n_neighbors: int = N_NEIGHBORS,
             random_state: int = RANDOM_STATE, target_ratio: float = 1.0, verbose: bool = True):
    """Return (X_res, y_res, is_synthetic, report).

    target_ratio: each minority class is grown to target_ratio x majority count (1.0 = fully balanced).
    """
    if verbose:
        step("A2. ADASYN - oversampling the training set only")
    X_train = X_train.reset_index(drop=True)
    y_train = pd.Series(np.asarray(y_train), name=y_train.name)
    counts = y_train.value_counts()
    majority = counts.idxmax()

    scaler = NumericScaler(spec).fit(X_train)       # fitted on the training set only
    Xs = scaler.transform(X_train)

    synth_X, synth_y, rows = [], [], []
    for cls, n in counts.items():
        if cls == majority:
            rows.append(dict(size_class=cls, n_original=n, k_used=None, n_synthetic=0, status="majority class"))
            continue
        target = int(round(counts[majority] * target_ratio))
        if n >= target:
            rows.append(dict(size_class=cls, n_original=n, k_used=None, n_synthetic=0, status="already at target"))
            continue
        if n < 2:
            rows.append(dict(size_class=cls, n_original=n, k_used=None, n_synthetic=0,
                             status="skipped: < 2 samples, cannot interpolate"))
            continue
        k = min(n_neighbors, n - 1)
        try:
            ada = ADASYN(sampling_strategy={cls: target}, n_neighbors=k, random_state=random_state)
            X_r, y_r = ada.fit_resample(Xs, y_train)
            new = X_r[len(Xs):]                     # imblearn appends the synthetic rows after the originals
            synth_X.append(new); synth_y += [cls] * len(new)
            status = "ok" if k == n_neighbors else f"ok (k reduced to {k}: small class)"
            rows.append(dict(size_class=cls, n_original=n, k_used=k, n_synthetic=len(new), status=status))
        except (ValueError, RuntimeError) as err:   # e.g. no minority point has a majority neighbour
            rows.append(dict(size_class=cls, n_original=n, k_used=k, n_synthetic=0, status=f"skipped: {err}"))

    report = pd.DataFrame(rows)
    report["n_after"] = report["n_original"] + report["n_synthetic"]

    if synth_X:
        synth = pd.DataFrame(scaler.inverse_transform(np.vstack(synth_X)), columns=X_train.columns)
        synth = _repair(synth, X_train, spec)
    else:
        synth = X_train.iloc[:0].copy()
    X_res = pd.concat([X_train, synth], ignore_index=True)
    y_res = pd.Series(list(y_train) + synth_y, name=y_train.name)
    is_synthetic = np.r_[np.zeros(len(X_train), bool), np.ones(len(synth), bool)]

    if verbose:
        print(f"ADASYN(n_neighbors={n_neighbors}, sampling_strategy=per class -> {target_ratio:.0%} of majority, "
              f"random_state={random_state})")
        show(report)
        print(f"Training rows: {len(X_train):,} -> {len(X_res):,} ({len(synth):,} synthetic)")
    return X_res, y_res, is_synthetic, report
