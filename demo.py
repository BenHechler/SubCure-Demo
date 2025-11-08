import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from Datasets import load_ACS_dataset, load_twins_dataset, load_german_dataset, load_SO_dataset, DATASET_META
from Algorithms import subcure_tuple, subcure_pattern, estimate_ate_linear

st.set_page_config(page_title="SubCure Demo", layout="wide")

defaults = {
    "df": None,
    "treatment_col": None,
    "outcome_col": None,
    "confounders": [],
    "desired_center": None,
    "tolerance": 0.5,
    "algorithm": "Tuple-level (remove individual rows)",
    "max_removals": 20,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
elif dataset_name == "Stack Overflow":
    df = load_SO_dataset()
elif dataset_name == "Credit":
    df = load_german_dataset()
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
# dataset_name is whatever the user picked: "Twins", "ACS", "German", "SO"
meta_key = dataset_name  # make sure names match the keys above

st.subheader("Dataset summary")
if meta_key in DATASET_META:
    meta = DATASET_META[meta_key]
    summary_df = pd.DataFrame(
        {
            "Dataset": [meta_key],
            "#Tuples": [meta["#tuples"]],
            "#Atts": [meta["#atts"]],
            "Treatment": [meta["treatment"]],
            "Outcome": [meta["outcome"]],
            "Confounding Variables": [meta["confounders"]],
            "Org ATE": [meta["org_ate"]],
            "Tar ATE": [meta["tar_ate"]],
        }
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No paper metadata for this dataset.")


# ---------------------------------------------------------------------
# 3) Column selection
# ---------------------------------------------------------------------
st.subheader("Step 2 – Select columns for causal estimation")

all_cols = list(df.columns)
treatment_col = st.selectbox("Treatment (binary 0/1)", all_cols)
outcome_col = st.selectbox("Outcome (numeric)", all_cols, index=min(1, len(all_cols) - 1))
confounders = st.multiselect(
    "Confounders (if exists)",
    [c for c in all_cols if c not in [treatment_col, outcome_col]],
    default=[],
)


if "ate_val" not in st.session_state:
    st.session_state.ate_val = None

if st.button("Compute ATE"):
    st.session_state.ate_val = estimate_ate_linear(df, treatment_col, outcome_col, confounders)

if st.session_state.ate_val is not None:
    st.success(f"Current ATE = **{st.session_state.ate_val:.4f}**")


# ---------------------------------------------------------------------
# 4) Target range, visualization, algorithm
# ---------------------------------------------------------------------
st.subheader("Step 3 – Define target range and repair algorithm")

current_ate = estimate_ate_linear(df, treatment_col, outcome_col, confounders)

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


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
st.subheader("Step 4 – Run repair")

if st.button("Run repair"):
    if algorithm.startswith("Tuple"):
        df_new, removed_df, new_ate = subcure_tuple(
            df,
            treatment_col,
            outcome_col,
            confounders,
            ate_target=desired_center,
            eps=tolerance,
            max_iters=st.session_state.max_removals,
        )
        st.success(f"New ATE after tuple-level repair: **{new_ate:.4f}**")
        st.write(f"Total tuples removed: **{len(removed_df)}**")
        st.subheader("Removed tuples")
        st.dataframe(removed_df.head(10))
    else:
        df_new, removed_df, new_ate = subcure_pattern(
            df,
            treatment_col,
            outcome_col,
            confounders,
            ate_target=desired_center,
            eps=tolerance,
            max_walks=st.session_state.max_removals,
        )
        st.success(f"New ATE after pattern-level repair: **{new_ate:.4f}**")
        st.write(f"Total patterns removed: **{len(removed_df)}**")
        st.subheader("Removed subpopulations")
        st.dataframe(removed_df.head(10))

    st.info(
        "Tuple-level removes individual rows.\n"
        "Pattern-level removes attribute=value groups (simplified version)."
    )
