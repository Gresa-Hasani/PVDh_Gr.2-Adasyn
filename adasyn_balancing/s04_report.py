"""Step 4 - Tables and figures for the comparison."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import ROOT, SIZE_CLASSES
from common.utils import save_csv, save_figure, show, step

OUT_ADASYN = ROOT / "outputs" / "adasyn"
SETTINGS = ["Baseline", "ADASYN full (1.0)", "ADASYN CV-selected"]
COLORS = dict(zip(SETTINGS, ["#2a78d6", "#eb6834", "#6250d6"]))


def class_distribution(y_train, resampled: dict, y_test) -> pd.DataFrame:
    """resampled: {column label: y of an ADASYN training set}."""
    cols = {"train_before_ADASYN": pd.Series(y_train).value_counts()}
    cols.update({k: pd.Series(v).value_counts() for k, v in resampled.items()})
    cols["test_original"] = pd.Series(y_test).value_counts()
    t = pd.DataFrame(cols).reindex(SIZE_CLASSES).fillna(0).astype(int)
    for c in list(t.columns):
        t[f"{c}_%"] = (t[c] / t[c].sum() * 100).round(3)
    t.index.name = "size_class"
    return t


def plot_distribution(dist: pd.DataFrame, path):
    cols = [c for c in dist.columns if not c.endswith("_%")]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 4.6), sharey=True)
    for ax, col in zip(axes, cols):
        ax.bar(SIZE_CLASSES, dist[col], color=["#9aa5b1", "#2a78d6", "#6250d6", "#eb6834"])
        ax.set_yscale("log"); ax.set_ylim(10, dist[cols].to_numpy().max() * 8)
        title = {"train_before_ADASYN": "Train - original", "test_original": "Test (never resampled)"}.get(
            col, "Train - " + col.replace("train_after_", "ADASYN "))
        ax.set_title(title, fontsize=10); ax.set_xlabel("size class")
        for i, (v, p) in enumerate(zip(dist[col], dist[f"{col}_%"])):
            ax.text(i, v, f"{v:,}\n{p:.2f}%", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("rows (log scale)")
    fig.suptitle("Class distribution (ADASYN rows beyond the original counts are synthetic)"); fig.tight_layout()
    save_figure(fig, path)


def plot_metrics(summary: pd.DataFrame, path):
    metrics = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
               "roc_auc_macro"]
    models = summary["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.8), sharey=True)
    x = np.arange(len(metrics)); w = 0.27
    for ax, m in zip(np.atleast_1d(axes), models):
        for j, setting in enumerate(SETTINGS):
            r = summary.query("model == @m and training_set == @setting").iloc[0]
            label = setting if setting != "ADASYN CV-selected" else f"ADASYN CV ({r['target_ratio']:.2f})"
            bars = ax.bar(x + (j - 1) * w, r[metrics].astype(float), w, label=label, color=COLORS[setting])
            ax.bar_label(bars, fmt="%.2f", fontsize=6)
        ax.set_xticks(x, [s.replace("_", "\n") for s in metrics], fontsize=8)
        ax.set_ylim(0, 1.1); ax.set_title(m); ax.legend(fontsize=7, loc="lower left")
    np.atleast_1d(axes)[0].set_ylabel("score on the test set")
    fig.suptitle("Baseline vs ADASYN - overall test metrics (macro = every class weighted equally)"); fig.tight_layout()
    save_figure(fig, path)


def plot_recall_precision(per_class: pd.DataFrame, path):
    models = per_class["model"].unique()
    fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 8), sharey=True)
    x = np.arange(len(SIZE_CLASSES)); w = 0.27
    for i, metric in enumerate(["recall", "precision"]):
        for k, m in enumerate(models):
            ax = axes[i, k]
            for j, setting in enumerate(SETTINGS):
                r = per_class.query("model == @m and training_set == @setting").set_index("size_class")
                bars = ax.bar(x + (j - 1) * w, r.loc[SIZE_CLASSES, metric], w, color=COLORS[setting], label=setting)
                ax.bar_label(bars, fmt="%.2f", fontsize=6)
            ax.set_xticks(x, SIZE_CLASSES); ax.set_ylim(0, 1.1); ax.set_title(f"{m} - {metric}")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Per-class recall (top) and precision (bottom) on the test set"); fig.tight_layout()
    save_figure(fig, path)


def plot_confusion(cms: dict, path):
    models = list(dict.fromkeys(m for m, _ in cms))
    fig, axes = plt.subplots(len(SETTINGS), len(models), figsize=(5.2 * len(models), 4.6 * len(SETTINGS)))
    for i, setting in enumerate(SETTINGS):
        for j, m in enumerate(models):
            cm = cms[(m, setting)]
            norm = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[i, j]
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
            for a in range(len(SIZE_CLASSES)):
                for b in range(len(SIZE_CLASSES)):
                    ax.text(b, a, f"{norm[a, b]:.2f}\n({cm[a, b]:,})", ha="center", va="center", fontsize=7,
                            color="white" if norm[a, b] > 0.5 else "black")
            ax.set_xticks(range(4), SIZE_CLASSES); ax.set_yticks(range(4), SIZE_CLASSES)
            ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{m} - {setting}", fontsize=10)
    fig.suptitle("Confusion matrices on the test set (row-normalised = recall; counts in brackets)")
    fig.tight_layout()
    save_figure(fig, path)


def plot_cv_sensitivity(grid: pd.DataFrame, chosen: dict, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, metric, title in [(axes[0], "f1_macro", "Macro F1 (selection criterion)"),
                              (axes[1], "balanced_accuracy", "Balanced accuracy"),
                              (axes[2], "accuracy", "Accuracy")]:
        for (m, g), col in zip(grid.groupby("model", sort=False), ["#2a78d6", "#eb6834", "#6250d6"]):
            ax.errorbar(g["target_ratio"], g[f"{metric}_mean"], yerr=g[f"{metric}_std"], marker="o", capsize=3,
                        color=col, label=m)
            if metric == "f1_macro":
                sel = g[g["target_ratio"] == chosen[m]]
                ax.scatter(sel["target_ratio"], sel[f"{metric}_mean"], s=160, facecolors="none", edgecolors=col, lw=2)
        ax.set_xlabel("ADASYN target_ratio (0 = no ADASYN)"); ax.set_title(title)
    axes[0].legend(fontsize=8); axes[0].set_ylabel("3-fold CV inside the training set (mean ± sd)")
    fig.suptitle("Sensitivity to target_ratio - cross-validation on the training set only (circled = selected)")
    fig.tight_layout()
    save_figure(fig, path)


def report(dist, gen_reports: pd.DataFrame, summary, per_class, cms, cv_grid, chosen):
    step("A4. RESULTS")
    OUT_ADASYN.mkdir(parents=True, exist_ok=True)
    print("Class distribution:"); show(dist)
    cols = ["model", "training_set", "target_ratio", "accuracy", "balanced_accuracy", "precision_macro",
            "recall_macro", "f1_macro", "f1_weighted", "roc_auc_macro", "roc_auc_weighted"]
    print("\nOverall metrics on the test set:"); show(summary[cols].round(3), n=None)
    print("\nPer-class metrics:"); show(per_class.round(3), n=None)

    save_csv(dist.reset_index(), OUT_ADASYN / "01_class_distribution.csv")
    save_csv(gen_reports, OUT_ADASYN / "02_adasyn_generation.csv")
    save_csv(summary.round(4), OUT_ADASYN / "03_metrics_overall.csv")
    save_csv(per_class.round(4), OUT_ADASYN / "04_metrics_per_class.csv")
    cm_rows = [{"model": m, "training_set": s, "true": SIZE_CLASSES[a], **dict(zip(SIZE_CLASSES, cm[a]))}
               for (m, s), cm in cms.items() for a in range(len(SIZE_CLASSES))]
    save_csv(pd.DataFrame(cm_rows), OUT_ADASYN / "05_confusion_matrices.csv")
    save_csv(cv_grid.round(4).assign(selected=[chosen[m] == r for m, r in zip(cv_grid["model"], cv_grid["target_ratio"])]),
             OUT_ADASYN / "07_cv_ratio_sensitivity.csv")

    plot_distribution(dist, OUT_ADASYN / "fig01_class_distribution.png")
    plot_metrics(summary, OUT_ADASYN / "fig02_metrics_baseline_vs_adasyn.png")
    plot_recall_precision(per_class, OUT_ADASYN / "fig03_per_class_recall_precision.png")
    plot_confusion(cms, OUT_ADASYN / "fig04_confusion_matrices.png")
    plot_cv_sensitivity(cv_grid, chosen, OUT_ADASYN / "fig05_cv_ratio_sensitivity.png")
