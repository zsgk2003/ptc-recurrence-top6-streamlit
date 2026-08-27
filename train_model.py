"""
Reproduce and export the LightGBM Top-6-feature model for PTC recurrence prediction.

The pipeline mirrors `model_LightGBMtop6_features.ipynb` exactly (same features,
same 70/30 stratified split with random_state=42, same GridSearchCV space and CV
folds), so the exported model matches the published results in
`thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM_top6_features`.

Usage:
    python train_model.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "modeldata_335_PTC.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

FEATURE_NAMES = ["Age", "Physical Examination", "Adenopathy", "T", "N", "Response"]
TARGET = "Recurred"

PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 50],
}

# Test-set metrics reported by the original notebook run, used as a regression check.
REFERENCE_TEST_METRICS = {
    "Accuracy": 0.9604,
    "Precision": 1.0,
    "Recall": 0.8519,
    "F1-score": 0.9200,
    "AUC": 0.9905,
    "Brier Score": 0.0364,
}


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
        "Brier Score": brier_score_loss(y_true, y_prob),
        "AP": average_precision_score(y_true, y_prob),
    }


def bootstrap_ci(y_true, y_prob, n_bootstrap: int = 1000) -> tuple[float, float]:
    rng = np.random.RandomState(RANDOM_STATE)
    y_true = np.asarray(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def fit_pipeline() -> dict:
    """Run the full pipeline in memory and return the model plus its metadata.

    The app calls this directly when no usable artifacts are on disk, e.g. on a
    fresh Streamlit Community Cloud deployment, so the deployed app never depends
    on a pickle staying loadable across library versions.
    """
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # LightGBM is trained on raw values; the scaler is exported only so that the
    # app can show standardized feature positions in the cohort.
    scaler = StandardScaler().fit(X_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    metrics = {}
    for split, X_split, y_split in (("train", X_train, y_train), ("test", X_test, y_test)):
        y_prob = model.predict_proba(X_split)[:, 1]
        y_pred = model.predict(X_split)
        metrics[split] = compute_metrics(y_split, y_pred, y_prob)
        metrics[split]["AUC_CI"] = bootstrap_ci(y_split, y_prob)

    train_set = X_train.copy()
    train_set[TARGET] = y_train.values

    test_set = X_test.copy()
    test_set[TARGET] = y_test.values
    test_set["Predicted_Prob"] = model.predict_proba(X_test)[:, 1]

    metadata = {
        "model": "LightGBM",
        "feature_names": FEATURE_NAMES,
        "best_params": grid.best_params_,
        "cv_auc": grid.best_score_,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "recurred_train": int(y_train.sum()),
        "recurred_test": int(y_test.sum()),
        "metrics": metrics,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return {
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
        "train_set": train_set,
        "test_set": test_set,
    }


def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    result = fit_pipeline()
    model, meta = result["model"], result["metadata"]

    print(f"Total {meta['n_total']} | train {meta['n_train']} "
          f"(recurred {meta['recurred_train']}) | test {meta['n_test']} "
          f"(recurred {meta['recurred_test']})")
    print(f"Best params: {meta['best_params']}")
    print(f"Best CV AUC: {meta['cv_auc']:.4f}")

    for split in ("train", "test"):
        print(f"\n[{split}] " + "  ".join(
            f"{k}={v:.4f}" for k, v in meta["metrics"][split].items() if k != "AUC_CI"
        ))

    print("\nRegression check against the published notebook run:")
    for key, expected in REFERENCE_TEST_METRICS.items():
        got = meta["metrics"]["test"][key]
        flag = "OK" if abs(got - expected) < 5e-3 else "MISMATCH"
        print(f"  {key:<13} expected {expected:.4f} | got {got:.4f}  [{flag}]")

    joblib.dump(model, os.path.join(ARTIFACT_DIR, "model_LightGBM_top6.pkl"))
    joblib.dump(result["scaler"], os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    result["train_set"].to_csv(os.path.join(ARTIFACT_DIR, "training_set.csv"), index=False)
    result["test_set"].to_csv(os.path.join(ARTIFACT_DIR, "testing_set.csv"), index=False)
    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nArtifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
