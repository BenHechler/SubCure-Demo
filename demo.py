import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from Datasets import load_ACS_dataset, load_twins_dataset
from Algorithms import compute_ate_linear, run_tuple_repair, run_pattern_repair

st.set_page_config(page_title="SubCure Demo", layout="wide")
st.title("SubCure Demo")

st.markdown(
    """
A demo for **“Stress-Testing Causal Claims via Cardinality Repairs”**

**Flow**
1. Choose dataset  
2. Select treatment, outcome, confounders and compute ATE  
3. Set target ATE range and repair algorithm (tuple-level or pattern-level)  
4. Run the repair and inspect removed data
"""
)

# ---------------------------------------------------------------------
# 1) Choose dataset
# ---------------------------------------------------------------------
st.subheader("Step 1 – Choose dataset")

dataset_name = st.selectbox("Select a dataset", ["Twins", "ACS", "Credit", "Stack Overflow", "Upload CSV"])

if dataset_name == "Twins":
    df = load_twins_dataset()
elif dataset_name == "ACS":
    df = load_ACS_dataset()
else:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.stop()
    df = pd.read_csv(uploaded)

# ---------------------------------------------------------------------
# 2) Show dataset preview
# ---------------------------------------------------------------------
st.subheader("Dataset preview")
st.dataframe(df.head())

# ---------------------------------------------------------------------
# 3) Column selection
# ---------------------------------------------------------------------
st.subheader("Step 2 – Select columns for causal estimation")

all_cols = list(df.columns)
treatment_col = st.selectbox("Treatment (binary 0/1)", all_cols)
outcome_col = st.selectbox("Outcome (numeric)", all_cols, index=min(1, len(all_cols) - 1))
confounders = st.multiselect(
    "Confounders (optional)",
    [c for c in all_cols if c not in [treatment_col, outcome_col]],
    default=[],
)

if st.button("Compute ATE"):
    ate_val = compute_ate_linear(df, treatment_col, outcome_col, confounders)
    st.success(f"Current ATE = **{ate_val:.4f}**")

# ---------------------------------------------------------------------
# 4) Target range, visualization, algorithm
# ---------------------------------------------------------------------
st.subheader("Step 3 – Define target range and repair algorithm")

current_ate = compute_ate_linear(df, treatment_col, outcome_col, confounders)

# two columns: controls on left, visualization on right
left, right = st.columns([1, 1])

with left:
    desired_center = st.number_input(
        "Desired ATE center",
        value=float(np.round(current_ate, 3))
    )
    tolerance = st.number_input(
        "Tolerance (±)",
        value=0.5,
        min_value=0.0,
        step=0.1
    )
    algorithm = st.selectbox(
        "Choose repair algorithm",
        ["Tuple-level (remove individual rows)", "Pattern-level (remove subpopulations)"],
    )
    max_removals = st.slider("Max iterations", 1, 100, 20)

with right:
    st.markdown("**Outcome by treatment (quick look)**")
    # >>> VISUALIZATION 1: outcome distribution by treatment <<<
    if df[treatment_col].nunique() <= 5:
        # small number of treatment groups → boxplot-like
        chart_data = df[[treatment_col, outcome_col]].rename(
            columns={treatment_col: "treatment", outcome_col: "outcome"}
        )
        box = alt.Chart(chart_data).mark_boxplot().encode(
            x=alt.X("treatment:O", title="Treatment"),
            y=alt.Y("outcome:Q", title=outcome_col),
        ).properties(height=200)
        st.altair_chart(box, use_container_width=True)
    else:
        st.write("Treatment has many values, skipping boxplot.")

    st.markdown("**Current vs target ATE**")
    # >>> VISUALIZATION 2: current ATE vs target band <<<
    ate_df = pd.DataFrame({
        "label": ["Current ATE", "Target min", "Target max"],
        "value": [current_ate, desired_center - tolerance, desired_center + tolerance],
    })
    base = alt.Chart(ate_df).mark_rule(color="lightgray").encode(x="value:Q")
    points = alt.Chart(ate_df).mark_point(size=80).encode(
        x="value:Q",
        y=alt.value(0),
        color=alt.Color("label:N", legend=alt.Legend(orient="bottom")),
        tooltip=["label", "value"]
    ).properties(height=80)
    st.altair_chart(base + points, use_container_width=True)


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
st.subheader("Step 4 – Run repair")

if st.button("Run repair"):
    if algorithm.startswith("Tuple"):
        df_new, removed_df, new_ate = run_tuple_repair(df, treatment_col, outcome_col, confounders, max_removals, desired_center, tolerance)
        st.success(f"New ATE after tuple-level repair: **{new_ate:.4f}**")
        st.write(f"Total tuples removed: **{len(removed_df)}**")
        st.subheader("Removed tuples")
        st.dataframe(removed_df.head(10))
    else:
        df_new, removed_df, new_ate = run_pattern_repair(df, treatment_col, outcome_col, confounders, max_removals, desired_center, tolerance)
        st.success(f"New ATE after pattern-level repair: **{new_ate:.4f}**")
        st.write(f"Total patterns removed: **{len(removed_df)}**")
        st.subheader("Removed subpopulations")
        st.dataframe(removed_df.head(10))

    st.info(
        "Tuple-level removes individual rows.\n"
        "Pattern-level removes attribute=value groups (simplified version)."
    )
