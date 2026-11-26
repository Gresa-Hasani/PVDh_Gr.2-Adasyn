"""Step 6 - Choosing a milder ADASYN ratio with cross-validation INSIDE the training set.

Full balancing (target_ratio = 1.0) multiplies the smallest class about 2,800 times. A milder ratio is often
better, but it must not be chosen on the test set: that would make the test results optimistic.

Procedure (per model, the test set is never touched):
    * stratified K-fold split of the TRAINING set (K = 3, seed 42)
    * for each fold and each ratio in RATIO_GRID: ADASYN is fitted on the K-1 training folds only; the
      validation fold is never resampled; the model is trained and scored on the validation fold
    * criterion: mean macro F1 over the folds. Macro F1 needs both recall AND precision of every class;
      balanced accuracy would always prefer the most aggressive ratio, because it ignores precision
    * the preferred ratio is the best ratio > 0 (ties -> the smaller ratio). Ratio 0 = no ADASYN is scored too,
      so it is visible whether ADASYN helps at all under this criterion

The full grid of CV scores is saved; it is the sensitivity analysis.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from adasyn_balancing import s02_adasyn, s03_models
from common.config import RANDOM_STATE
from common.utils import show, step

RATIO_GRID = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]
N_FOLDS = 3
CRITERION = "f1_macro"


def cross_validate_ratios(ds, ratios=RATIO_GRID, n_folds: int = N_FOLDS, random_state: int = RANDOM_STATE):
    step(f"A6. RATIO SELECTION - {n_folds}-fold stratified CV inside the training set (test set not used)")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    X, y = ds.X_train.reset_index(drop=True), ds.y_train.reset_index(drop=True)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr, X_va, y_va = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]
        for ratio in ratios:
            if ratio == 0:
                X_fit, y_fit = X_tr, y_tr
            else:                                    # ADASYN sees the training folds only
                X_fit, y_fit, _, _ = s02_adasyn.resample(X_tr, y_tr, ds.spec, target_ratio=ratio,
                                                         random_state=random_state, verbose=False)
            for name, model in s03_models.make_models(random_state).items():
                model.fit(X_fit, y_fit)
                r, pc, _ = s03_models.evaluate(model, X_va, y_va)
                rec = pc.set_index("size_class")["recall"]
                rows.append({"model": name, "target_ratio": ratio, "fold": fold, "n_train": len(X_fit),
                             **{k: r[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
                                                  "precision_macro", "roc_auc_macro")},
                             **{f"recall_{c}": rec[c] for c in rec.index}})
        print(f"  fold {fold}/{n_folds} done")
    folds = pd.DataFrame(rows)
    metrics = [c for c in folds.columns if c not in ("model", "target_ratio", "fold", "n_train")]
    grid = folds.groupby(["model", "target_ratio"], sort=False)[metrics].agg(["mean", "std"])
    grid.columns = [f"{m}_{s}" for m, s in grid.columns]
    grid = grid.reset_index()

    chosen = {}
    for name, g in grid.groupby("model", sort=False):
        cand = g[g["target_ratio"] > 0].sort_values([f"{CRITERION}_mean", "target_ratio"], ascending=[False, True])
        best = cand.iloc[0]
        no_adasyn = g.loc[g["target_ratio"] == 0, f"{CRITERION}_mean"].iloc[0]
        chosen[name] = float(best["target_ratio"])
        print(f"  {name:<20} preferred ratio {best['target_ratio']:.2f}: CV macro F1 "
              f"{best[f'{CRITERION}_mean']:.3f} ± {best[f'{CRITERION}_std']:.3f} (no ADASYN: {no_adasyn:.3f})")
    show(grid[["model", "target_ratio", "balanced_accuracy_mean", "f1_macro_mean", "f1_macro_std",
               "accuracy_mean", "roc_auc_macro_mean"]].round(3), n=None)
    return grid, chosen
