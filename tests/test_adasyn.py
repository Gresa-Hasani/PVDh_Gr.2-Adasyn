"""Unit tests for the ADASYN step (standard-library unittest, no extra dependency).

Run from the repository root:  python -m unittest discover -s tests -v
They use small synthetic data, so they do not need data/processed/.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adasyn_balancing import s01_dataset, s02_adasyn  # noqa: E402
from adasyn_balancing.s01_dataset import FeatureSpec  # noqa: E402


def make_data(sizes, seed=0):
    """Overlapping classes with mixed feature types, like the real model matrix."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, (cls, n) in enumerate(sizes.items()):
        m = rng.poisson(1 + i, n); f = rng.poisson(0.3 + 0.3 * i, n)
        cat = rng.choice(3, n, p=[0.6 - 0.1 * i, 0.3, 0.1 + 0.1 * i])
        X.append(pd.DataFrame({
            "age_years": rng.normal(10 + i, 4, n).clip(0),
            "Total M": m.astype(float), "Total F": f.astype(float), "owners_total": (m + f).astype(float),
            "has_female_owner": (f > 0).astype(float),
            "form_A": (cat == 0).astype(float), "form_B": (cat == 1).astype(float), "form_C": (cat == 2).astype(float),
        }))
        y += [cls] * n
    return pd.concat(X, ignore_index=True), pd.Series(y, name="Madhësia")


SPEC = FeatureSpec(continuous=["age_years"], integer=["Total M", "Total F", "owners_total"],
                   binary=["has_female_owner"], onehot_groups={"form": ["form_A", "form_B", "form_C"]})


class TestAdasynResample(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_data({"0-9": 600, "10-49": 60, "50-249": 20})

    def run_resample(self, X=None, y=None, **kw):
        return s02_adasyn.resample(self.X if X is None else X, self.y if y is None else y, SPEC, verbose=False, **kw)

    def test_balances_minority_classes(self):
        X_res, y_res, syn, rep = self.run_resample()
        counts = y_res.value_counts()
        for cls in ("10-49", "50-249"):            # ADASYN is approximately, not exactly, balanced
            self.assertGreater(counts[cls], 0.8 * counts["0-9"])
        self.assertEqual(counts["0-9"], 600)        # majority untouched

    def test_original_rows_unchanged_and_first(self):
        X_res, y_res, syn, _ = self.run_resample()
        n = len(self.X)
        self.assertFalse(syn[:n].any())
        self.assertTrue(np.array_equal(X_res.iloc[:n].to_numpy(), self.X.to_numpy()))
        self.assertEqual(list(y_res[:n]), list(self.y))

    def test_feature_structure_preserved(self):
        X_res, _, syn, _ = self.run_resample()
        s = X_res[syn]
        self.assertEqual(list(X_res.columns), list(self.X.columns))
        self.assertTrue((s[["form_A", "form_B", "form_C"]].sum(axis=1) == 1).all())
        self.assertTrue(s["has_female_owner"].isin([0, 1]).all())
        for c in ("Total M", "Total F", "owners_total"):
            self.assertTrue(np.array_equal(s[c], s[c].round()))
        self.assertTrue(np.allclose(s["owners_total"], s["Total M"] + s["Total F"]))
        self.assertTrue(((s["Total F"] > 0) == (s["has_female_owner"] == 1)).all())

    def test_reproducible(self):
        a, *_ = self.run_resample(random_state=7)
        b, *_ = self.run_resample(random_state=7)
        self.assertTrue(a.equals(b))

    def test_small_class_reduces_k(self):
        X, y = make_data({"0-9": 600, "10-49": 60, "250-": 3})
        _, y_res, _, rep = self.run_resample(X, y)
        row = rep.set_index("size_class").loc["250-"]
        self.assertEqual(row["k_used"], 2)
        self.assertGreater(row["n_synthetic"], 0)

    def test_single_sample_class_is_skipped(self):
        X, y = make_data({"0-9": 600, "10-49": 60, "250-": 1})
        _, y_res, _, rep = self.run_resample(X, y)
        row = rep.set_index("size_class").loc["250-"]
        self.assertEqual(row["n_synthetic"], 0)
        self.assertIn("skipped", row["status"])
        self.assertEqual((y_res == "250-").sum(), 1)

    def test_partial_target_ratio(self):
        _, y_res, _, _ = self.run_resample(target_ratio=0.5)
        self.assertLess(y_res.value_counts()["10-49"], 0.7 * 600)


class TestNumericOnlyScaling(unittest.TestCase):
    """Correction 1: rare one-hot / binary levels must not dominate the k-NN distances."""

    def setUp(self):
        self.X, _ = make_data({"0-9": 600, "10-49": 60})
        self.X.loc[0, ["form_A", "form_B", "form_C"]] = [0.0, 0.0, 1.0]    # keep the data valid
        self.X["rare_flag"] = 0.0
        self.X.loc[:1, "rare_flag"] = 1.0                                  # 2 rows of 660, like section U
        self.spec = FeatureSpec(continuous=["age_years"], integer=["Total M", "Total F", "owners_total"],
                                binary=["has_female_owner", "rare_flag"],
                                onehot_groups={"form": ["form_A", "form_B", "form_C"]})

    def test_binary_and_onehot_columns_stay_0_1(self):
        A = s02_adasyn.NumericScaler(self.spec).fit(self.X).transform(self.X)
        cols = list(self.X.columns)
        for c in ["has_female_owner", "rare_flag", "form_A", "form_B", "form_C"]:
            self.assertTrue(np.array_equal(A[:, cols.index(c)], self.X[c].to_numpy()))
        self.assertEqual(A[:, cols.index("rare_flag")].max(), 1.0)     # full standardisation would give ~18
        for c in ["age_years", "Total M"]:                               # numeric columns are standardised
            self.assertAlmostEqual(A[:, cols.index(c)].mean(), 0.0, places=9)

    def test_inverse_is_exact(self):
        sc = s02_adasyn.NumericScaler(self.spec).fit(self.X)
        self.assertTrue(np.allclose(sc.inverse_transform(sc.transform(self.X)), self.X.to_numpy(), atol=1e-12))

    def test_interpolation_commutes_with_scaling(self):
        sc = s02_adasyn.NumericScaler(self.spec).fit(self.X)
        A = sc.transform(self.X); a, b, lam = A[3], A[40], 0.37
        mid_scaled = sc.inverse_transform((a + lam * (b - a))[None, :])[0]
        x, z = self.X.iloc[3].to_numpy(), self.X.iloc[40].to_numpy()
        self.assertTrue(np.allclose(mid_scaled, x + lam * (z - x)))


class TestRegistrationAgeRepair(unittest.TestCase):
    """Correction 2: synthetic age may not exceed the time since the (repaired) registration date."""

    SPEC = FeatureSpec(continuous=["age_years"], integer=["Years", "registration_month"], binary=[],
                       onehot_groups={})

    @staticmethod
    def make(sizes, seed=3):
        rng = np.random.default_rng(seed)
        rows, y = [], []
        for cls, n in sizes.items():
            year = rng.integers(2000, 2027, n); month = rng.integers(1, 13, n)
            month = np.where(year == 2026, rng.integers(1, 10, n), month)
            age = s02_adasyn.max_age_years(pd.Series(year), pd.Series(month)).to_numpy()  # active: age = maximum
            closed = rng.random(n) < 0.3
            age = np.where(closed, age * rng.uniform(0, 1, n), age)
            rows.append(pd.DataFrame({"age_years": age, "Years": year.astype(float),
                                      "registration_month": month.astype(float)}))
            y += [cls] * n
        return pd.concat(rows, ignore_index=True), pd.Series(y, name="Madhësia")

    def test_joint_rounding_and_age_cap(self):
        X, _ = self.make({"a": 50})
        synth = pd.DataFrame({"age_years": [25.0, 1.5], "Years": [2025.0, 2005.5], "registration_month": [10.0, 6.5]})
        out = s02_adasyn._repair_registration_date(synth.copy(), X)
        # 2025-10 -> at most 2025-10-01..2026-10-05 = 1.01 years
        self.assertAlmostEqual(out.loc[0, "age_years"], (pd.Timestamp("2026-10-05") - pd.Timestamp("2025-10-01")).days / 365.25)
        # 2005.5 / 6.5 is month index 2005*12 + 11.5 -> about 2005-12/2006-01, NOT 2006-06 (separate rounding)
        self.assertTrue((out.loc[1, "Years"], out.loc[1, "registration_month"]) in {(2005.0, 12.0), (2006.0, 1.0)})
        self.assertEqual(out.loc[1, "age_years"], 1.5)                 # feasible ages are not changed

    def test_no_impossible_synthetic_ages(self):
        X, y = self.make({"0-9": 800, "10-49": 60, "50-249": 15})
        X_res, y_res, syn, _ = s02_adasyn.resample(X, y, self.SPEC, verbose=False)
        s = X_res[syn]
        over = s["age_years"] - s02_adasyn.max_age_years(s["Years"], s["registration_month"])
        self.assertGreater(len(s), 0)
        self.assertFalse((over > 1e-9).any())
        self.assertTrue(s["registration_month"].between(1, 12).all())


class TestDatasetSplit(unittest.TestCase):
    def test_split_before_preprocessing_and_disjoint(self):
        rng = np.random.default_rng(1)
        n = 2000
        df = pd.DataFrame({
            "Madhësia": rng.choice(["0-9", "10-49", "50-249", "250-"], n, p=[0.9, 0.06, 0.03, 0.01]),
            "age_years": np.where(rng.random(n) < 0.05, np.nan, rng.normal(10, 3, n)),
            "businesses_per_1000_municipality": rng.normal(150, 20, n),
            "Total M": rng.poisson(1, n), "Total F": rng.poisson(0.3, n), "Years": rng.integers(2000, 2027, n),
            "registration_month": rng.integers(1, 13, n), "has_trade_name": rng.integers(0, 2, n),
            "re_registered": 0, "legal_form_group": rng.choice(["Individual", "Ortakëri"], n),
            "status": rng.choice(["Regjistruar", "Pasiv"], n), "section_code": rng.choice(list("ABG"), n),
            "region": rng.choice(["Prizren", "Pejë"], n),
            "Nr. punëtorëve": rng.integers(0, 300, n), "log_employees": 0.0, "flag_employees_imputed": 0,
        })
        df["owners_total"] = df["Total M"] + df["Total F"]
        df["has_female_owner"] = (df["Total F"] > 0).astype(int)
        ds = s01_dataset.build(df)
        self.assertTrue(ds.X_train.index.intersection(ds.X_test.index).empty)
        self.assertFalse(set(s01_dataset.LEAKAGE_EXCLUDED) & set(ds.X_train.columns))
        # medians come from the training rows only
        self.assertAlmostEqual(ds.medians["age_years"], df.loc[ds.X_train.index, "age_years"].median())
        self.assertFalse(ds.X_test.isna().any().any())


if __name__ == "__main__":
    unittest.main()
