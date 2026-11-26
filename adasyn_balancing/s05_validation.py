"""Step 5 - Leakage and consistency checks. Any failed check stops the run (AssertionError)."""
import numpy as np
import pandas as pd

from common.utils import show, step
from adasyn_balancing.s01_dataset import LEAKAGE_EXCLUDED, TARGET
from adasyn_balancing.s02_adasyn import max_age_years
from common.columns import C


def validate(df_clean, ds, X_res, y_res, is_synthetic, label: str = "ADASYN") -> pd.DataFrame:
    """Run the checks for one ADASYN training set; returns the table (the caller saves all tables together)."""
    step(f"A5. VALIDATION - leakage and consistency checks ({label})")
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    n_tr = len(ds.X_train)
    check("Train and test rows are disjoint", ds.X_train.index.intersection(ds.X_test.index).empty,
          f"{n_tr:,} train / {len(ds.X_test):,} test")
    check("Every class is present in both sets", set(ds.y_train) == set(ds.y_test))
    check("Test set was not resampled (same size and class counts as after the split)",
          ds.y_test.value_counts().equals(df_clean.loc[ds.X_test.index, TARGET].value_counts()))
    check("Original training rows are unchanged inside the ADASYN set",
          np.array_equal(X_res.iloc[:n_tr].to_numpy(), ds.X_train.to_numpy())
          and list(y_res.iloc[:n_tr]) == list(ds.y_train))
    check("Synthetic rows only belong to minority classes",
          ds.y_train.value_counts().idxmax() not in set(y_res[is_synthetic]))

    test_keys = set(map(tuple, ds.X_test.to_numpy().round(6)))
    synth = X_res[is_synthetic]
    dup = sum(tuple(r) in test_keys for r in synth.to_numpy().round(6)) if len(synth) else 0
    dup_train = sum(tuple(r) in test_keys for r in ds.X_train.to_numpy().round(6))
    check("Synthetic rows that coincide with a test row (feature vector)", True,
          f"informational: {dup:,} of {len(synth):,} synthetic ({dup / max(len(synth), 1):.1%}) vs "
          f"{dup_train:,} of {n_tr:,} ORIGINAL train rows ({dup_train / n_tr:.1%}). Features are mostly discrete, "
          f"so equal vectors occur naturally; ADASYN is fitted without the test set")

    check("No employee-based feature is used", not set(LEAKAGE_EXCLUDED) & set(X_res.columns), str(LEAKAGE_EXCLUDED))
    check("Same feature columns, same order (train / test / ADASYN)",
          list(ds.X_train.columns) == list(ds.X_test.columns) == list(X_res.columns))
    check("No missing values in the model matrices", not (X_res.isna().any().any() or ds.X_test.isna().any().any()))
    onehot_ok = all((synth[cols].sum(axis=1) == 1).all() for cols in ds.spec.onehot_groups.values())
    check("Synthetic one-hot groups have exactly one 1", onehot_ok)
    check("Synthetic binary / integer columns are whole numbers",
          all(np.array_equal(synth[c], synth[c].round()) for c in ds.spec.binary + ds.spec.integer))
    check("Synthetic owners_total = Total M + Total F",
          np.allclose(synth["owners_total"], synth["Total M"] + synth["Total F"]))

    over = synth["age_years"] - max_age_years(synth[C.YEAR], synth["registration_month"])
    check("Synthetic age <= time from registration date to the reference date", not (over > 1e-9).any(),
          f"{int((over > 1e-9).sum()):,} violations (real rows: within Part I's 2-decimal rounding)")

    res = pd.DataFrame(checks).assign(training_set=label)
    show(res[["check", "passed", "detail"]], n=None)
    failed = res.loc[~res["passed"], "check"].tolist()
    assert not failed, f"Validation failed: {failed}"
    print("All checks passed.")
    return res
