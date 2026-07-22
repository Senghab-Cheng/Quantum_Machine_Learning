"""Streamlit frontend for the diabetes classical-vs-quantum ML comparison.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from inference import RAW_FEATURE_COLUMNS, fit_pipeline, transform_sample
from models.classical_models import (
    MODEL_INFO,
    evaluate_all_models,
    get_all_models,
    train_all_models,
    tune_all_models,
)
from theme import (
    CSS,
    COLORS,
    icon_alert,
    icon_brand,
    icon_capsule,
    icon_check,
    icon_clipboard,
    icon_clock,
    icon_folder,
    icon_test_tube,
)

st.set_page_config(page_title="Diabetes QML Comparison", page_icon="🩺", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading data and fitting preprocessing pipeline...")
def get_pipeline():
    return fit_pipeline()


@st.cache_resource(show_spinner="Training models (classical + quantum)...")
def get_trained_models(_pipeline):
    models = get_all_models()
    trained, training_info = train_all_models(models, _pipeline["X_train"], _pipeline["y_train"])
    results = evaluate_all_models(trained, _pipeline["X_test"], _pipeline["y_test"], training_info)
    return trained, results


@st.cache_resource(show_spinner="Tuning hyperparameters (grid search)...")
def get_tuned_models(_pipeline):
    tuned, best_params = tune_all_models(_pipeline["X_train"], _pipeline["y_train"], verbose=False)
    _, training_info = train_all_models(tuned, _pipeline["X_train"], _pipeline["y_train"])
    results = evaluate_all_models(tuned, _pipeline["X_test"], _pipeline["y_test"], training_info)
    return tuned, results, best_params


def results_to_df(results: dict) -> pd.DataFrame:
    rows = {
        name: {
            "Accuracy": r["Accuracy"], "Precision": r["Precision"],
            "Recall": r["Recall"], "F1 Score": r["F1 Score"],
            "Training Time (s)": r.get("Training Time", np.nan),
        }
        for name, r in results.items()
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "Model"
    return df


def grouped_metric_chart(df: pd.DataFrame, metrics: list) -> alt.Chart:
    """Side-by-side (not stacked) bars comparing each model across `metrics`."""
    long_df = df[metrics].reset_index().melt("Model", var_name="Metric", value_name="Score")
    return (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title=None, axis=None),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(range=[COLORS["dark"], COLORS["primary"], COLORS["light"], "#9BD4F5"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            column=alt.Column("Model:N", title=None, header=alt.Header(labelFontSize=13, labelFontWeight="bold")),
            tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".4f")],
        )
        .properties(width=140, height=280)
    )


def stat_card(icon_svg: str, value: str, label: str) -> str:
    return f"""
    <div class="stat-card">
      {icon_svg}
      <div><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>
    </div>
    """


st.markdown(
    f"""
    <div class="med-hero">
      {icon_brand(52)}
      <div>
        <h1>Diabetes Risk Comparison</h1>
        <p>Pima Indians Diabetes dataset &middot; Classical ML baselines &middot;
        Quantum models (VQC / QSVM)</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

pipeline = get_pipeline()
trained_models, base_results = get_trained_models(pipeline)

selected = option_menu(
    menu_title=None,
    options=["Predict", "Compare Models", "Dataset"],
    icons=["clipboard2-pulse", "bar-chart-line", "folder2-open"],
    orientation="horizontal",
    styles={
        "container": {
            "padding": "6px", "background-color": "#FFFFFF", "border-radius": "14px",
            "box-shadow": "0 4px 14px rgba(15,41,66,0.06)", "margin-bottom": "20px",
        },
        "icon": {"color": COLORS["primary"], "font-size": "16px"},
        "nav-link": {"font-size": "0.92rem", "text-align": "center", "margin": "0 4px", "border-radius": "10px"},
        "nav-link-selected": {"background-color": COLORS["primary"], "color": "white"},
    },
)

# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
if selected == "Predict":
    st.markdown(
        f'<h3 style="display:flex;align-items:center;gap:10px;">{icon_clipboard(28)} Patient Vitals</h3>',
        unsafe_allow_html=True,
    )

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.slider("Glucose", 0, 200, 117)
            blood_pressure = st.slider("Blood Pressure", 0, 130, 72)
            skin_thickness = st.slider("Skin Thickness", 0, 100, 23)
        with col2:
            insulin = st.slider("Insulin", 0, 850, 30)
            bmi = st.slider("BMI", 0.0, 70.0, 32.0, step=0.1)
            dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.37, step=0.01)
            age = st.slider("Age", 21, 90, 29)

        model_name = st.selectbox("Model", list(trained_models.keys()))
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        raw_input = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age,
        }
        X_sample = transform_sample(raw_input, pipeline)
        model = trained_models[model_name]
        prediction = model.predict(X_sample)[0]
        proba = model.predict_proba(X_sample)[0][1] if hasattr(model, "predict_proba") else None
        proba_text = f"Estimated probability: <b>{proba:.1%}</b>" if proba is not None else "This model doesn't expose class probabilities."

        if prediction == 1:
            st.markdown(
                f"""<div class="alert-banner risk">{icon_alert(34)}
                <div><p class="alert-title">Elevated diabetes risk</p>
                <p class="alert-sub">{proba_text} &middot; model: {model_name}</p></div></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="alert-banner clear">{icon_check(34)}
                <div><p class="alert-title">Low diabetes risk</p>
                <p class="alert-sub">{proba_text} &middot; model: {model_name}</p></div></div>""",
                unsafe_allow_html=True,
            )
        if proba is not None:
            st.progress(float(proba))

# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------
elif selected == "Compare Models":
    base_df = results_to_df(base_results)
    best_acc_name = base_df["Accuracy"].idxmax()
    fastest_name = base_df["Training Time (s)"].idxmin()
    best_f1_name = base_df["F1 Score"].idxmax()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(stat_card(icon_capsule(30), f"{base_df.loc[best_acc_name, 'Accuracy']:.1%}", f"Best accuracy &middot; {best_acc_name}"), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_card(icon_clock(30), f"{base_df.loc[fastest_name, 'Training Time (s)']:.3f}s", f"Fastest training &middot; {fastest_name}"), unsafe_allow_html=True)
    with c3:
        st.markdown(stat_card(icon_test_tube(30), f"{base_df.loc[best_f1_name, 'F1 Score']:.3f}", f"Best F1 score &middot; {best_f1_name}"), unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<h3 style="display:flex;align-items:center;gap:10px;">{icon_clipboard(26)} Model comparison (classical + quantum)</h3>',
        unsafe_allow_html=True,
    )
    st.dataframe(base_df.style.format("{:.4f}").highlight_max(
        subset=["Accuracy", "Precision", "Recall", "F1 Score"], color="#DCEEFC"
    ), width='stretch')
    st.altair_chart(grouped_metric_chart(base_df, ["Accuracy", "Precision", "Recall", "F1 Score"]))

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    if st.button("Run hyperparameter tuning for classical models (GridSearchCV, ~10-20s)"):
        tuned_models, tuned_results, best_params = get_tuned_models(pipeline)
        tuned_df = results_to_df(tuned_results)

        st.markdown(
            f'<h3 style="display:flex;align-items:center;gap:10px;">{icon_test_tube(26)} Tuned model comparison</h3>',
            unsafe_allow_html=True,
        )
        st.dataframe(tuned_df.style.format("{:.4f}").highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1 Score"], color="#DCEEFC"
        ), width='stretch')

        st.markdown("**Best parameters found**")
        for name, info in best_params.items():
            st.write(f"**{name}**: `{info['best_params']}` (CV accuracy = {info['best_score']:.4f})")

    with st.expander("What each model is"):
        for name, info in MODEL_INFO.items():
            st.markdown(f"**{name}** ({info['type']}) — {info['description']}")

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
else:
    st.markdown(
        f'<h3 style="display:flex;align-items:center;gap:10px;">{icon_folder(28)} Raw dataset</h3>',
        unsafe_allow_html=True,
    )
    st.dataframe(pipeline["raw_df"].head(20), width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Outcome distribution")
        st.bar_chart(pipeline["raw_df"]["Outcome"].value_counts().rename({0: "No Diabetes", 1: "Diabetes"}))
    with col2:
        st.caption("Summary statistics")
        st.dataframe(pipeline["raw_df"][RAW_FEATURE_COLUMNS].describe().T, width='stretch')
