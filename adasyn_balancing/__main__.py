"""ADASYN - runs all steps in order.   Usage:  python -m adasyn_balancing   (after python -m part1_preprocessing)

    A1 dataset -> A2 ADASYN (full balancing, reference) -> A6 ratio selection by CV inside the training set
    -> A2 ADASYN with each selected ratio -> A5 validation of every ADASYN training set
    -> A3 models on the same test set -> A4 tables and figures
"""
import time

import pandas as pd

from adasyn_balancing import s01_dataset, s02_adasyn, s03_models, s04_report, s05_validation, s06_ratio_selection
from common.utils import save_csv

FULL_RATIO = 1.0


def main():
    t0 = time.time()
    df = s01_dataset.load_clean()
    ds = s01_dataset.build(df)

    # ADASYN on the whole training set, once per distinct ratio (full balancing is kept as the reference)
    resampled, gen_reports = {}, []

    def adasyn(ratio):
        if ratio not in resampled:
            X_res, y_res, syn, rep = s02_adasyn.resample(ds.X_train, ds.y_train, ds.spec, target_ratio=ratio)
            resampled[ratio] = (X_res, y_res, syn)
            gen_reports.append(rep.assign(target_ratio=ratio))
        return resampled[ratio]

    adasyn(FULL_RATIO)
    cv_grid, chosen = s06_ratio_selection.cross_validate_ratios(ds)     # training data only
    for ratio in sorted(set(chosen.values())):
        adasyn(ratio)

    s04_report.OUT_ADASYN.mkdir(parents=True, exist_ok=True)
    checks = [s05_validation.validate(df, ds, X, y, syn, label=f"ADASYN ratio {r}")   # stops on any failure
              for r, (X, y, syn) in sorted(resampled.items())]
    save_csv(pd.concat(checks, ignore_index=True), s04_report.OUT_ADASYN / "06_validation_checks.csv")

    names = list(s03_models.make_models())
    training_sets = {
        "Baseline": {m: (ds.X_train, ds.y_train, 0.0) for m in names},
        "ADASYN full (1.0)": {m: (*resampled[FULL_RATIO][:2], FULL_RATIO) for m in names},
        "ADASYN CV-selected": {m: (*resampled[chosen[m]][:2], chosen[m]) for m in names},
    }
    summary, per_class, cms = s03_models.run(ds, training_sets)

    dist = s04_report.class_distribution(
        ds.y_train, {f"train_after_ratio_{r}": resampled[r][1] for r in sorted(resampled)}, ds.y_test)
    s04_report.report(dist, pd.concat(gen_reports, ignore_index=True), summary, per_class, cms, cv_grid, chosen)
    print(f"\nDONE: ADASYN finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
