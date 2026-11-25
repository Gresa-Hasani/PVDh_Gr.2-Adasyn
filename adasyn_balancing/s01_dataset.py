"""Step 1 - Dataset: target, features, train/test split and preprocessing.

Target:   `Madhësia` (4 ordered EU size classes, kept unchanged).
Features: only columns that are NOT derived from the employee count. Excluded for leakage:
          `Nr. punëtorëve`, `log_employees`, `flag_employees_imputed` (and min-max / z-score versions).
Excluded for redundancy: `is_active` / `is_closed` (= status one-hot), `flag_nace_missing` (= section "unknown"),
          `female_owner_share` (undefined when there are no owners), free text and identifiers.

Order of operations (prevents leakage):
    1. split the RAW rows 70/30, stratified by the target
    2. fit the preprocessing (median imputation, category lists) on the training rows only
    3. apply it to both sets; categories unseen in training become an all-zero one-hot row
"""
from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split

from common.columns import C
from common.config import CLEAN_CSV, RANDOM_STATE, SIZE_CLASSES
from common.utils import show, step

TARGET = C.SIZE
TEST_SIZE = 0.30

# Feature groups. The group decides how ADASYN's synthetic values are repaired (s02_adasyn).
CONTINUOUS = ["age_years", "businesses_per_1000_municipality"]
INTEGER = [C.OWNERS_M, C.OWNERS_F, "owners_total", C.YEAR, "registration_month"]
BINARY = ["has_female_owner", "has_trade_name", "re_registered"]
CATEGORICAL = ["legal_form_group", "status", "section_code", "region"]
LEAKAGE_EXCLUDED = [C.EMPLOYEES, "log_employees", "flag_employees_imputed"]


@dataclass
class FeatureSpec:
    """Column names of the model matrix, by type (needed to repair synthetic rows)."""
    continuous: list = field(default_factory=list)
    integer: list = field(default_factory=list)
    binary: list = field(default_factory=list)
    onehot_groups: dict = field(default_factory=dict)   # categorical column -> its one-hot columns

    @property
    def columns(self):
        return (self.continuous + self.integer + self.binary
                + [c for cols in self.onehot_groups.values() for c in cols])


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    spec: FeatureSpec
    medians: pd.Series            # imputation values, learned from the training set


def load_clean() -> pd.DataFrame:
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"{CLEAN_CSV} is missing - run `python -m part1_preprocessing` first.")
    return pd.read_csv(CLEAN_CSV, low_memory=False)


def _fit_preprocessing(train: pd.DataFrame):
    medians = train[CONTINUOUS + INTEGER].median()
    categories = {c: sorted(train[c].astype(str).unique()) for c in CATEGORICAL}
    return medians, categories


def _transform(raw: pd.DataFrame, medians, categories, spec: FeatureSpec) -> pd.DataFrame:
    parts = [raw[CONTINUOUS + INTEGER].astype(float).fillna(medians),
             raw[BINARY].astype(float)]
    for col in CATEGORICAL:
        cat = pd.Categorical(raw[col].astype(str), categories=categories[col])   # unseen -> NaN -> all zeros
        parts.append(pd.get_dummies(cat, prefix=col, dtype=float).set_axis(raw.index))
    X = pd.concat(parts, axis=1)
    return X[spec.columns]


def build(df: pd.DataFrame | None = None, test_size: float = TEST_SIZE,
          random_state: int = RANDOM_STATE) -> Dataset:
    step("A1. DATASET - target, features, stratified train/test split")
    df = load_clean() if df is None else df
    df = df[df[TARGET].notna()]
    y = pd.Series(pd.Categorical(df[TARGET], categories=SIZE_CLASSES, ordered=True), index=df.index, name=TARGET)

    # 1. split first (raw rows), stratified so that every class appears in both sets
    train_raw, test_raw, y_train, y_test = train_test_split(
        df, y, test_size=test_size, stratify=y, random_state=random_state)

    # 2. preprocessing learned on train only
    medians, categories = _fit_preprocessing(train_raw)
    spec = FeatureSpec(CONTINUOUS.copy(), INTEGER.copy(), BINARY.copy(),
                       {c: [f"{c}_{v}" for v in categories[c]] for c in CATEGORICAL})

    # 3. same transformation for both sets
    X_train = _transform(train_raw, medians, categories, spec)
    X_test = _transform(test_raw, medians, categories, spec)

    print(f"Rows: {len(df):,} -> train {len(X_train):,} / test {len(X_test):,} (stratified, seed {random_state})")
    print(f"Features: {X_train.shape[1]} ({len(CONTINUOUS)} continuous, {len(INTEGER)} integer, "
          f"{len(BINARY)} binary, {len(spec.columns) - len(CONTINUOUS + INTEGER + BINARY)} one-hot "
          f"from {len(CATEGORICAL)} categorical)")
    print("Excluded for leakage (derived from the employee count):", LEAKAGE_EXCLUDED)
    show(pd.DataFrame({"train": y_train.value_counts().reindex(SIZE_CLASSES),
                       "test": y_test.value_counts().reindex(SIZE_CLASSES)}))
    return Dataset(X_train, X_test, y_train.astype(str), y_test.astype(str), spec, medians)
