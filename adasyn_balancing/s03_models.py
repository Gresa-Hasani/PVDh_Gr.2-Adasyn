"""Step 3 - Baseline vs ADASYN: the same models, the same test set.

The repository had no classification models before this module, so three standard scikit-learn classifiers
are used. Every setting is identical in all runs; only the training rows differ:
    Baseline            - the original training set
    ADASYN full (1.0)   - every minority class grown to the majority size (reference experiment)
    ADASYN CV-selected  - the ratio chosen per model by cross-validation inside the training set (s06)
No `class_weight` is used, so the effect measured is the effect of ADASYN alone.

Metrics (multiclass): accuracy, balanced accuracy, precision / recall / F1 (macro and weighted), ROC-AUC one-vs-rest
(macro and weighted, from predict_proba), per-class precision / recall / F1 and the confusion matrix.
Macro averages weight every class equally and therefore show the minority classes; weighted averages and accuracy
are dominated by "0-9" (98% of the rows).
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common.config import RANDOM_STATE, SIZE_CLASSES
from common.utils import step


def make_models(random_state: int = RANDOM_STATE) -> dict:
    """Fresh, unfitted models - called once per training set so both runs start identical."""
    return {
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000, random_state=random_state)),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=-1,
                                                random_state=random_state),
    }


def evaluate(model, X_test, y_test, labels=SIZE_CLASSES):
    pred = model.predict(X_test)
    # roc_auc_score needs the labels in sorted (string) order, with the probability columns in the same order
    auc_labels = sorted(labels)
    proba = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_).reindex(columns=auc_labels, fill_value=0)
    row = {"accuracy": accuracy_score(y_test, pred), "balanced_accuracy": balanced_accuracy_score(y_test, pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=labels, average=avg, zero_division=0)
        row.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
        row[f"roc_auc_{avg}"] = roc_auc_score(y_test, proba.to_numpy(), multi_class="ovr", average=avg,
                                              labels=auc_labels)
    p, r, f, s = precision_recall_fscore_support(y_test, pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"size_class": labels, "precision": p, "recall": r, "f1": f, "support": s})
    cm = confusion_matrix(y_test, pred, labels=labels)
    return row, per_class, cm


def run(ds, training_sets: dict):
    """training_sets: {setting label: {model name: (X_train, y_train, target_ratio)}}."""
    step("A3. MODELS - baseline vs ADASYN (same split, features, settings, test set)")
    summary, per_class, cms = [], [], {}
    for setting, per_model in training_sets.items():
        for name, model in make_models().items():
            X_tr, y_tr, ratio = per_model[name]
            model.fit(X_tr, y_tr)
            row, pc, cm = evaluate(model, ds.X_test, ds.y_test)
            summary.append({"model": name, "training_set": setting, "target_ratio": ratio, "n_train": len(X_tr), **row})
            per_class.append(pc.assign(model=name, training_set=setting, target_ratio=ratio))
            cms[(name, setting)] = cm
            print(f"  {name:<20} {setting:<20} ratio {ratio:<4} balanced acc {row['balanced_accuracy']:.3f} | "
                  f"macro F1 {row['f1_macro']:.3f} | accuracy {row['accuracy']:.3f}")
    summary = pd.DataFrame(summary)
    per_class = pd.concat(per_class, ignore_index=True)[["model", "training_set", "target_ratio", "size_class",
                                                         "precision", "recall", "f1", "support"]]
    return summary, per_class, cms
