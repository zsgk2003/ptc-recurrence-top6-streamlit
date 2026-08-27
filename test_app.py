"""
Headless end-to-end smoke test of the Streamlit app.

Exercises every page and the prediction flow through Streamlit's AppTest harness,
so regressions surface without opening a browser.

Usage:
    python test_app.py
"""

from __future__ import annotations

import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from streamlit.testing.v1 import AppTest

from feature_schema import FEATURE_NAMES

PAGES = ["Single Prediction", "Batch Prediction", "Model Performance", "About"]
TIMEOUT = 120


def new_app(page: str) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, f"exception on load: {at.exception}"
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"exception on page '{page}': {at.exception}"
    return at


def check_pages() -> None:
    for page in PAGES:
        at = new_app(page)
        print(f"  page '{page}': OK "
              f"({len(at.markdown)} markdown, {len(at.dataframe)} dataframes)")


def check_low_risk_prediction() -> float:
    at = new_app("Single Prediction")
    at.button[0].click().run()
    assert not at.exception, f"exception after predict: {at.exception}"
    metric_labels = [m.label for m in at.metric]
    assert "Recurrence probability" in metric_labels, metric_labels
    prob = at.metric[0].value
    print(f"  default (low-risk) profile -> probability {prob}, "
          f"risk category '{at.metric[1].value}'")
    return prob


def check_high_risk_prediction() -> float:
    at = new_app("Single Prediction")
    at.number_input[0].set_value(70)
    # selectbox order: Physical Examination, Adenopathy, T, N, Response
    at.selectbox[3].set_value("N1b")
    at.selectbox[4].set_value("Structural Incomplete")
    at.button[0].click().run()
    assert not at.exception, f"exception after predict: {at.exception}"
    prob = at.metric[0].value
    print(f"  high-risk profile -> probability {prob}, "
          f"risk category '{at.metric[1].value}'")
    return prob


def check_threshold_slider() -> None:
    at = new_app("Model Performance")
    at.sidebar.slider[0].set_value(0.25).run()
    assert not at.exception, f"exception after threshold change: {at.exception}"
    labels = [m.label for m in at.metric]
    assert "Sensitivity" in labels or "AUC" in labels, labels
    print("  threshold slider at 0.25: OK")


def check_batch_scoring() -> None:
    """Score the exported test set the same way the batch page does."""
    model = joblib.load(os.path.join("artifacts", "model_LightGBM_top6.pkl"))
    test = pd.read_csv(os.path.join("artifacts", "testing_set.csv"))
    probs = model.predict_proba(test[FEATURE_NAMES])[:, 1]
    auc = roc_auc_score(test["Recurred"], probs)
    accuracy = float(((probs >= 0.5).astype(int) == test["Recurred"].values).mean())
    assert abs(auc - 0.9905) < 5e-4, f"test AUC drifted: {auc:.4f}"
    assert abs(accuracy - 0.9604) < 5e-4, f"test accuracy drifted: {accuracy:.4f}"
    print(f"  batch scoring of {len(test)} test cases: AUC {auc:.4f}, "
          f"accuracy {accuracy:.4f}")


def main() -> int:
    print("Checking all pages render...")
    check_pages()

    print("Checking single-patient prediction...")
    low = float(check_low_risk_prediction().rstrip("%"))
    high = float(check_high_risk_prediction().rstrip("%"))
    assert high > low, f"high-risk probability {high} not above low-risk {low}"

    print("Checking batch scoring...")
    check_batch_scoring()

    print("Checking threshold slider...")
    check_threshold_slider()

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
