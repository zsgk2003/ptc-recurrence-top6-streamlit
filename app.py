"""
Papillary Thyroid Carcinoma (PTC) recurrence risk predictor.

Model: LightGBM trained on the Top-6 ablation-selected features
(Age, Physical Examination, Adenopathy, T, N, Response).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from feature_schema import (
    AGE_DEFAULT,
    AGE_RANGE,
    CATEGORICAL_OPTIONS,
    CODE_TO_LABEL,
    DEFAULT_OPTION_INDEX,
    FEATURE_HELP,
    FEATURE_LABELS,
    FEATURE_NAMES,
    display_value,
    encoding_table,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")

RISK_BANDS = [
    (0.20, "Low", "#16a34a"),
    (0.50, "Low-Moderate", "#eab308"),
    (0.75, "Moderate-High", "#f97316"),
    (1.01, "High", "#dc2626"),
]

st.set_page_config(
    page_title="PTC Recurrence Predictor | LightGBM Top-6",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1250px;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      .hero {
        background: linear-gradient(120deg, #0f766e 0%, #0891b2 100%);
        color: #fff; padding: 22px 28px; border-radius: 14px; margin-bottom: 22px;
      }
      .hero h1 {margin: 0; font-size: 1.7rem; font-weight: 700;}
      .hero p {margin: 8px 0 0; opacity: .93; font-size: .94rem; line-height: 1.6;}
      .pill {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: .78rem; font-weight: 600; background: #e0f2fe; color: #0369a1;
        margin-right: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Cached resources
# ============================================================
@st.cache_resource(show_spinner="Training the model (first run only)...")
def load_bundle() -> dict:
    """Model, metadata and the test split, from disk when possible.

    Falls back to re-running the training pipeline in memory, which keeps a fresh
    deployment working even when no artifacts were committed or the pickle cannot
    be read by the installed library versions.
    """
    model_path = os.path.join(ARTIFACT_DIR, "model_LightGBM_top6.pkl")
    meta_path = os.path.join(ARTIFACT_DIR, "metadata.json")
    test_path = os.path.join(ARTIFACT_DIR, "testing_set.csv")

    if all(os.path.exists(p) for p in (model_path, meta_path, test_path)):
        try:
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
            return {
                "model": joblib.load(model_path),
                "metadata": metadata,
                "test_set": pd.read_csv(test_path),
                "source": "artifacts",
            }
        except Exception as exc:  # noqa: BLE001 - any load failure falls back to training
            st.warning(f"Could not load the exported model ({exc}); retraining from data.")

    from train_model import fit_pipeline

    result = fit_pipeline()
    return {
        "model": result["model"],
        "metadata": result["metadata"],
        "test_set": result["test_set"],
        "source": "retrained",
    }


@st.cache_resource(show_spinner=False)
def load_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data(show_spinner=False)
def load_cohort() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "modeldata_335_PTC.csv"))


@st.cache_data(show_spinner=False)
def load_model_ranking():
    path = os.path.join(DATA_DIR, "model_ranking_top6.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0)


# ============================================================
# Helpers
# ============================================================
def risk_band(prob: float) -> tuple[str, str]:
    for upper, label, color in RISK_BANDS:
        if prob < upper:
            return label, color
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[FEATURE_NAMES])[:, 1]


def shap_values_for(explainer, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP values (log-odds) for the positive class, plus the base value."""
    values = explainer.shap_values(X[FEATURE_NAMES])
    base = explainer.expected_value
    if isinstance(values, list):
        values, base = values[1], base[1]
    elif np.ndim(values) == 3:
        values, base = values[:, :, 1], base[1]
    if isinstance(base, (list, np.ndarray)):
        base = float(np.ravel(base)[-1])
    return np.asarray(values), float(base)


def gauge_figure(prob: float, color: str, threshold: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 44, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.75},
            "steps": [
                {"range": [0, 20], "color": "#dcfce7"},
                {"range": [20, 50], "color": "#fef9c3"},
                {"range": [50, 75], "color": "#ffedd5"},
                {"range": [75, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "#334155", "width": 3},
                "thickness": 0.85,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=10))
    return fig


def contribution_figure(values: np.ndarray, row: pd.Series) -> go.Figure:
    order = np.argsort(np.abs(values))
    labels, texts, colors = [], [], []
    for i in order:
        feature = FEATURE_NAMES[i]
        shown = display_value(feature, row[feature])
        labels.append(
            f"{FEATURE_LABELS[feature]}<br>"
            f"<span style='font-size:11px;color:#64748b'>{shown}</span>"
        )
        texts.append(f"{values[i]:+.2f}")
        colors.append("#dc2626" if values[i] > 0 else "#2563eb")

    fig = go.Figure(go.Bar(
        x=values[order], y=labels, orientation="h",
        marker_color=colors, text=texts, textposition="outside",
        hovertemplate="SHAP = %{x:.3f}<extra></extra>",
    ))
    span = max(float(np.abs(values).max()), 0.5) * 1.5
    fig.update_layout(
        height=380, margin=dict(l=10, r=20, t=30, b=30),
        xaxis_title="SHAP value (log-odds); > 0 increases recurrence risk",
        xaxis_range=[-span, span], showlegend=False, plot_bgcolor="white",
    )
    fig.add_vline(x=0, line_width=1.5, line_color="#94a3b8")
    return fig


# ============================================================
# Load resources and build the sidebar
# ============================================================
bundle = load_bundle()
model = bundle["model"]
meta = bundle["metadata"]
test_df = bundle["test_set"]
explainer = load_explainer(model)
cohort = load_cohort()

test_metrics = meta.get("metrics", {}).get("test", {})

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Page",
    ["Single Prediction", "Batch Prediction", "Model Performance", "About"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Decision threshold", 0.05, 0.95, 0.50, 0.05,
    help="A predicted probability at or above this value is classified as recurrence. "
         "0.50 reproduces the published results.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Deployed model")
if test_metrics:
    ci = test_metrics.get("AUC_CI") or [None, None]
    ci_txt = f" (95% CI {ci[0]:.3f}-{ci[1]:.3f})" if ci[0] is not None else ""
    st.sidebar.success(
        f"**LightGBM - Top 6 features**\n\n"
        f"- Test AUC: **{test_metrics['AUC']:.4f}**{ci_txt}\n"
        f"- Accuracy: **{test_metrics['Accuracy']:.4f}**\n"
        f"- F1-score: **{test_metrics['F1-score']:.4f}**\n"
        f"- Brier score: **{test_metrics['Brier Score']:.4f}**"
    )
st.sidebar.caption(
    f"{meta.get('n_total', '-')} PTC patients | "
    f"train {meta.get('n_train', '-')} / test {meta.get('n_test', '-')}"
)

st.markdown(
    """
    <div class="hero">
      <h1>Papillary Thyroid Carcinoma Recurrence Risk Predictor</h1>
      <p>A <b>LightGBM</b> model built on a cohort of 335 papillary thyroid carcinoma (PTC)
      patients, using the <b>6 core clinical features</b> retained by a stepwise ablation
      study (age, physical examination, adenopathy, T stage, N stage, response to therapy).
      Every prediction comes with an individual SHAP explanation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Page 1: single prediction
# ============================================================
if page == "Single Prediction":
    st.subheader("Patient characteristics")

    with st.form("single_case"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input(
                FEATURE_LABELS["Age"], min_value=AGE_RANGE[0], max_value=AGE_RANGE[1],
                value=AGE_DEFAULT, step=1, help=FEATURE_HELP["Age"],
            )
            physical = st.selectbox(
                FEATURE_LABELS["Physical Examination"],
                list(CATEGORICAL_OPTIONS["Physical Examination"]),
                index=DEFAULT_OPTION_INDEX["Physical Examination"],
                help=FEATURE_HELP["Physical Examination"],
            )
        with c2:
            adenopathy = st.selectbox(
                FEATURE_LABELS["Adenopathy"],
                list(CATEGORICAL_OPTIONS["Adenopathy"]),
                index=DEFAULT_OPTION_INDEX["Adenopathy"],
                help=FEATURE_HELP["Adenopathy"],
            )
            t_stage = st.selectbox(
                FEATURE_LABELS["T"], list(CATEGORICAL_OPTIONS["T"]),
                index=DEFAULT_OPTION_INDEX["T"], help=FEATURE_HELP["T"],
            )
        with c3:
            n_stage = st.selectbox(
                FEATURE_LABELS["N"], list(CATEGORICAL_OPTIONS["N"]),
                index=DEFAULT_OPTION_INDEX["N"], help=FEATURE_HELP["N"],
            )
            response = st.selectbox(
                FEATURE_LABELS["Response"], list(CATEGORICAL_OPTIONS["Response"]),
                index=DEFAULT_OPTION_INDEX["Response"], help=FEATURE_HELP["Response"],
            )

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        record = {
            "Age": int(age),
            "Physical Examination": CATEGORICAL_OPTIONS["Physical Examination"][physical],
            "Adenopathy": CATEGORICAL_OPTIONS["Adenopathy"][adenopathy],
            "T": CATEGORICAL_OPTIONS["T"][t_stage],
            "N": CATEGORICAL_OPTIONS["N"][n_stage],
            "Response": CATEGORICAL_OPTIONS["Response"][response],
        }
        X_one = pd.DataFrame([record])
        prob = float(predict_proba(model, X_one)[0])
        label, color = risk_band(prob)
        positive = prob >= threshold

        st.markdown("---")
        st.subheader("Prediction")

        left, right = st.columns([1, 1.25])
        with left:
            st.plotly_chart(gauge_figure(prob, color, threshold))
        with right:
            m1, m2 = st.columns(2)
            m1.metric("Recurrence probability", f"{prob * 100:.2f}%")
            m2.metric("Risk category", label)
            advice = (
                "Consider shorter follow-up intervals, closer neck ultrasound and "
                "thyroglobulin monitoring, and multidisciplinary review of whether more "
                "aggressive treatment is warranted."
                if prob >= 0.5 else
                "Routine surveillance is appropriate: scheduled neck ultrasound and "
                "serum Tg / TgAb monitoring."
            )
            st.markdown(
                f"""
                <div style="background:{color}1a;border-left:6px solid {color};
                            padding:16px 18px;border-radius:10px;margin-top:6px;">
                  <div style="font-size:1.03rem;font-weight:700;color:{color};">
                    Classified as {'RECURRENCE' if positive else 'NO RECURRENCE'}
                    (threshold {threshold:.2f})
                  </div>
                  <div style="margin-top:8px;font-size:.92rem;line-height:1.7;color:#334155;">
                    {advice}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Note: on this small, well-separated cohort the boosted model produces "
                "highly saturated probabilities - most patients score close to 0% or "
                "100%. Read the output as a risk category rather than a finely "
                "calibrated absolute probability."
            )

        st.markdown("#### Individual explanation (SHAP)")
        values, base = shap_values_for(explainer, X_one)
        total = float(values[0].sum())
        st.caption(
            f"Cohort baseline log-odds {base:.2f}; feature contributions sum to "
            f"{total:+.2f}, giving {base + total:.2f} for this patient. "
            "Red bars push the risk up, blue bars pull it down."
        )
        st.plotly_chart(contribution_figure(values[0], X_one.iloc[0]))

        with st.expander("Encoded model input and cohort comparison"):
            comp = pd.DataFrame({
                "Feature": [FEATURE_LABELS[f] for f in FEATURE_NAMES],
                "This patient": [display_value(f, record[f]) for f in FEATURE_NAMES],
                "Encoded value": [record[f] for f in FEATURE_NAMES],
                "Cohort median": [float(cohort[f].median()) for f in FEATURE_NAMES],
            })
            st.dataframe(comp, use_container_width=True, hide_index=True)

        st.warning(
            "**Disclaimer**: for research and education only. This output cannot replace "
            "professional medical diagnosis; clinical decisions must be made by a "
            "qualified physician after a complete evaluation."
        )


# ============================================================
# Page 2: batch prediction
# ============================================================
elif page == "Batch Prediction":
    st.subheader("Batch prediction")
    st.markdown(
        "Upload a CSV file to score many patients at once. The file must contain these "
        "**6 columns** with integer codes (see the About page for the encoding scheme):"
    )
    st.code(", ".join(FEATURE_NAMES), language="text")

    template = pd.DataFrame([{
        "Age": 45, "Physical Examination": 2, "Adenopathy": 3,
        "T": 0, "N": 0, "Response": 0,
    }])
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download CSV template", data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="ptc_top6_template.csv", mime="text/csv", use_container_width=True,
    )
    c2.download_button(
        f"Download test-set example ({len(test_df)} cases)",
        data=test_df[FEATURE_NAMES + ["Recurred"]].to_csv(index=False).encode("utf-8-sig"),
        file_name="ptc_top6_testset_example.csv", mime="text/csv",
        use_container_width=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            st.stop()

        missing = [c for c in FEATURE_NAMES if c not in df_in.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        st.success(f"Loaded {len(df_in)} records")
        st.dataframe(df_in.head(10), use_container_width=True)

        probs = predict_proba(model, df_in)
        preds = (probs >= threshold).astype(int)

        result = df_in.copy()
        result["Recurrence_Probability"] = np.round(probs, 4)
        result["Prediction"] = np.where(preds == 1, "Recurrence", "No recurrence")
        result["Risk_Category"] = [risk_band(p)[0] for p in probs]

        st.markdown("#### Results")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records", len(result))
        k2.metric("Predicted recurrence", int(preds.sum()))
        k3.metric("Predicted no recurrence", int((preds == 0).sum()))
        k4.metric("Mean probability", f"{probs.mean() * 100:.2f}%")

        if "Recurred" in df_in.columns and df_in["Recurred"].nunique() > 1:
            auc = roc_auc_score(df_in["Recurred"], probs)
            acc = float((preds == df_in["Recurred"].values).mean())
            st.info(
                f"The upload contains ground-truth labels (`Recurred`): "
                f"AUC = **{auc:.4f}**, accuracy = **{acc:.4f}** at threshold {threshold:.2f}."
            )

        st.dataframe(result, use_container_width=True)

        fig = go.Figure(go.Histogram(x=probs, nbinsx=20, marker_color="#0891b2"))
        fig.add_vline(x=threshold, line_dash="dash", line_color="#dc2626",
                      annotation_text=f"threshold {threshold:.2f}")
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
            xaxis_title="Predicted recurrence probability", yaxis_title="Patients",
        )
        st.plotly_chart(fig)

        st.download_button(
            "Download results (CSV)",
            data=result.to_csv(index=False).encode("utf-8-sig"),
            file_name="ptc_top6_predictions.csv", mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# Page 3: model performance
# ============================================================
elif page == "Model Performance":
    st.subheader("Model performance")

    if test_metrics:
        cols = st.columns(6)
        for col, key in zip(
            cols, ["AUC", "Accuracy", "Precision", "Recall", "F1-score", "Brier Score"]
        ):
            col.metric(key, f"{test_metrics[key]:.4f}")
        st.caption(
            f"Held-out test set: {meta.get('n_test')} patients "
            f"({meta.get('recurred_test')} with recurrence). "
            f"Best hyper-parameters `{meta.get('best_params')}`, "
            f"5-fold CV AUC {meta.get('cv_auc', 0):.4f}."
        )

    if test_df is not None and "Recurred" in test_df.columns:
        y_true = test_df["Recurred"].values
        y_prob = predict_proba(model, test_df)
        y_pred = (y_prob >= threshold).astype(int)

        tab_roc, tab_pr, tab_cm, tab_shap = st.tabs(
            ["ROC curve", "PR curve", "Confusion matrix", "Global SHAP"]
        )

        with tab_roc:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"LightGBM (AUC = {roc_auc_score(y_true, y_prob):.4f})",
                line=dict(color="#0f766e", width=3), fill="tozeroy",
                fillcolor="rgba(15,118,110,0.12)",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                line=dict(color="#94a3b8", dash="dash"),
            ))
            fig.update_layout(
                height=460, plot_bgcolor="white",
                xaxis_title="1 - specificity (false positive rate)",
                yaxis_title="Sensitivity (true positive rate)",
                legend=dict(x=0.45, y=0.08), margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig)

        with tab_pr:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            fig = go.Figure(go.Scatter(
                x=rec, y=prec, mode="lines", line=dict(color="#b45309", width=3),
                fill="tozeroy", fillcolor="rgba(180,83,9,0.12)", name="LightGBM",
            ))
            fig.add_hline(
                y=float(y_true.mean()), line_dash="dash", line_color="#94a3b8",
                annotation_text=f"prevalence {y_true.mean():.3f}",
            )
            fig.update_layout(
                height=460, plot_bgcolor="white", xaxis_title="Recall",
                yaxis_title="Precision", margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig)

        with tab_cm:
            cm = confusion_matrix(y_true, y_pred)
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Predicted: no recurrence", "Predicted: recurrence"],
                y=["Actual: no recurrence", "Actual: recurrence"],
                text=cm, texttemplate="%{text}", colorscale="Teal", showscale=False,
            ))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_autorange="reversed")
            st.plotly_chart(fig)
            tn, fp, fn, tp = cm.ravel()
            s1, s2, s3 = st.columns(3)
            s1.metric("Sensitivity", f"{tp / (tp + fn):.4f}")
            s2.metric("Specificity", f"{tn / (tn + fp):.4f}")
            s3.metric("Threshold", f"{threshold:.2f}")

        with tab_shap:
            values, _ = shap_values_for(explainer, test_df)
            importance = pd.DataFrame({
                "Feature": [FEATURE_LABELS[f] for f in FEATURE_NAMES],
                "Mean |SHAP|": np.abs(values).mean(axis=0),
            }).sort_values("Mean |SHAP|")
            fig = go.Figure(go.Bar(
                x=importance["Mean |SHAP|"], y=importance["Feature"], orientation="h",
                marker_color="#0891b2",
                text=importance["Mean |SHAP|"].round(3), textposition="outside",
            ))
            fig.update_layout(
                height=420, plot_bgcolor="white",
                xaxis_title="Mean absolute SHAP value (test set)",
                margin=dict(l=10, r=40, t=30, b=10),
            )
            st.plotly_chart(fig)
            st.caption(
                "Response to therapy dominates, followed by N stage and age - "
                "consistent with the SHAP analysis of the original study."
            )

    ranking = load_model_ranking()
    if ranking is not None:
        st.markdown("#### All 9 models on the same Top-6 feature set (test set)")
        show = ranking[["AUC", "Accuracy", "Precision", "Recall", "F1-score", "Brier Score"]]
        st.dataframe(
            show.style.format("{:.4f}")
            .background_gradient(subset=["AUC", "Accuracy", "F1-score"], cmap="Greens")
            .background_gradient(subset=["Brier Score"], cmap="Reds_r"),
            use_container_width=True,
        )
        st.caption("LightGBM has the highest AUC on this feature set and is the model deployed here.")


# ============================================================
# Page 4: about
# ============================================================
else:
    st.subheader("About this model")

    st.markdown(
        """
        <span class="pill">LightGBM</span><span class="pill">6 features</span>
        <span class="pill">335 PTC patients</span><span class="pill">SHAP explanations</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Background
        Differentiated thyroid cancer is the most common endocrine malignancy. Overall
        prognosis is favourable, yet a substantial minority of patients recur after
        treatment, so reliable recurrence-risk estimates matter for individualised
        treatment and surveillance planning.

        ### Data and modelling
        - **Source**: UCI Machine Learning Repository - Differentiated Thyroid Cancer
          Recurrence (`Thyroid_Diff.csv`, 383 patients), restricted to the
          **335 patients with histologically confirmed papillary thyroid carcinoma**.
        - **Outcome**: recurrence during follow-up (90 events, 26.87%).
        - **Split**: stratified 70/30 into 234 training and 101 test patients,
          `random_state=42`.
        - **Feature selection**: stepwise ablation over the 16 clinical variables ranked by
          LightGBM importance; the **Top-6 subset** preserves discrimination while cutting
          model complexity and the amount of data that has to be collected.
        - **Tuning**: GridSearchCV with 5-fold stratified cross-validation, optimising AUC.
        - **Interpretability**: SHAP TreeExplainer, both global importance and per-patient
          contribution breakdowns.

        ### Deployed model
        """
    )

    if test_metrics:
        perf = pd.DataFrame([
            {"Metric": key, "Test set": f"{test_metrics[key]:.4f}"}
            for key in ["AUC", "Accuracy", "Precision", "Recall", "F1-score", "Brier Score"]
        ])
        st.dataframe(perf, use_container_width=True, hide_index=True)
        st.caption(f"Best hyper-parameters: `{meta.get('best_params')}`")

    st.markdown("### Features and encoding scheme")
    st.dataframe(pd.DataFrame(encoding_table()), use_container_width=True, hide_index=True)

    st.markdown("### Cohort distribution")
    feature_pick = st.selectbox(
        "Feature", FEATURE_NAMES, format_func=lambda f: FEATURE_LABELS[f]
    )
    fig = go.Figure()
    for value, name, color in ((0, "No recurrence", "#0891b2"), (1, "Recurrence", "#dc2626")):
        subset = cohort.loc[cohort["Recurred"] == value, feature_pick]
        if feature_pick == "Age":
            fig.add_trace(go.Histogram(x=subset, name=name, marker_color=color,
                                       opacity=0.75, nbinsx=25))
        else:
            counts = subset.value_counts().sort_index()
            fig.add_trace(go.Bar(
                x=[display_value(feature_pick, c) for c in counts.index],
                y=counts.values, name=name, marker_color=color,
            ))
    fig.update_layout(
        barmode="overlay" if feature_pick == "Age" else "group",
        height=380, plot_bgcolor="white", yaxis_title="Patients",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig)

    st.markdown(
        """
        ### Limitations and disclaimer
        - The model was developed on a single public cohort of 335 patients and has not been
          validated in an external cohort.
        - Response to therapy is assessed **after** initial treatment, so this tool is a
          post-treatment risk re-assessment aid rather than a pre-operative predictor.
        - Predicted probabilities are strongly saturated (in the test set, 96 of 101
          patients fall below 0.01 or above 0.99). Discrimination is excellent, but the
          numeric probability should not be over-interpreted as a calibrated risk.
        - For research and education only. It cannot replace professional medical diagnosis;
          all clinical decisions must be made by a qualified physician after a complete
          clinical evaluation.
        """
    )

    if meta:
        with st.expander("Model metadata (metadata.json)"):
            st.json(meta)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:.82rem;'>"
    "PTC recurrence risk predictor | LightGBM on 6 clinical features | "
    "research and educational use only"
    "</div>",
    unsafe_allow_html=True,
)
