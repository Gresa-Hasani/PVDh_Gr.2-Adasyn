"""
ADASYN - class-imbalance handling for the business size class (Madhësia)

Input:   data/processed/arbk_clean.csv   (output of Part I, `python -m part1_preprocessing`)
Output:  outputs/adasyn/                 (tables + figures)

Task:    multiclass classification of the EU size class "0-9" / "10-49" / "50-249" / "250-"
         WITHOUT any employee-based feature (the size class is computed from the employee count,
         so those columns would leak the target).

Modules:
    s01_dataset     - target, features, stratified train/test split, preprocessing fitted on train only
    s02_adasyn      - ADASYN on the training set only (numeric-only scaling, guards for tiny classes,
                      one-hot / integer / registration-date repair)
    s03_models      - the same classifiers trained on the original vs the ADASYN training set
    s04_report      - comparison tables and figures
    s05_validation  - leakage and consistency checks (the run stops if any check fails)
    s06_ratio_selection - milder ADASYN ratios, chosen by stratified CV inside the training set only

Compared on one test set: Baseline | ADASYN full (1.0, reference) | ADASYN with the CV-selected ratio.

Run:  python -m adasyn_balancing          (needs `python -m part1_preprocessing` first)
Tests: python -m unittest discover -s tests
"""
