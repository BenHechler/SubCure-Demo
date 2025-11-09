import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from Datasets import load_ACS_dataset, load_twins_dataset, load_german_dataset, load_SO_dataset, DATASET_META
from Algorithms import subcure_tuple, subcure_pattern, estimate_ate_linear
import time
st.set_page_config(page_title="SubCure Demo", layout="wide")

# ---------------------- SESSION STATE DEFAULTS ----------------------
defaults = {
    "df": None,
    "ate_val": None,
    "desired_center": None,
    "tolerance": 0.5,
    "algorithm": "Tuple-level (remove individual rows)",
    "max_removals": 20,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------- SIDEBAR CONFIG ----------------------
if "new_ate" not in st.session_state:
    st.session_state.new_ate = None

st.sidebar.title("⚙️ Controls")

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Dataset",
    ["Twins", "ACS", "Credit", "Stack Overflow", "Upload CSV"]
)

# Load dataset
if dataset_name == "Twins":
    df = load_twins_dataset()
elif dataset_name == "ACS":
    df = load_ACS_dataset()
elif dataset_name == "Credit":
    df = load_german_dataset()
elif dataset_name == "Stack Overflow":
    df = load_SO_dataset()
else:
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.stop()
    df = pd.read_csv(uploaded)

st.session_state.df = df

all_cols = list(df.columns)
treatment_col = st.sidebar.selectbox("Treatment (binary 0/1)", all_cols)
outcome_col = st.sidebar.selectbox("Outcome (numeric)", [c for c in all_cols if c not in [treatment_col]])
confounders = st.sidebar.multiselect(
    "Confounders",
    [c for c in all_cols if c not in [treatment_col, outcome_col]],
    default=[c for c in all_cols if c not in [treatment_col, outcome_col]]
)

if "ate_val" not in st.session_state:
    st.session_state.ate_val = None

if st.sidebar.button("Compute ATE"):
    st.session_state.ate_val = estimate_ate_linear(df, treatment_col, outcome_col, confounders)

if st.session_state.ate_val is not None:
    st.sidebar.markdown(
        f"**Current ATE:** `{st.session_state.ate_val:.4f}`"
    )
else:
    st.sidebar.info("Click **Compute ATE** to calculate it.")


current_ate = estimate_ate_linear(df, treatment_col, outcome_col, confounders)
desired_center = st.sidebar.number_input(
    "Desired ATE center",
    value=float(np.round(current_ate, 3))
)

tolerance = st.sidebar.slider(
    "Tolerance (±)",
    min_value=0.0,
    max_value=5.0,
    value=st.session_state.tolerance,
    step=0.1,
    help="Allowed deviation around the target ATE"
)

algorithm = st.sidebar.selectbox(
    "Repair algorithm",
    ["Tuple-level (remove individual rows)", "Pattern-level (remove subpopulations)"]
)


if "run_repair" not in st.session_state:
    st.session_state.run_repair = False

if st.sidebar.button("Run repair"):
    st.session_state.run_repair = True

if st.session_state.new_ate is not None:
    st.sidebar.markdown(
        f"**New ATE after repair:** `{st.session_state.new_ate:.4f}`"
    )


# ---------------------- MAIN PANEL ----------------------
st.title("SubCure Demo – Stress Testing Causal Claims")

col1, col2 = st.columns([1.4, 1])

# ===== LEFT SIDE: Dataset and Results =====
with col1:
    st.markdown("### 📊 Dataset preview")
    st.dataframe(df.head(), use_container_width=True, height=180)

    st.markdown("### 📘 Dataset summary")
    meta_key = dataset_name
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
        st.table(summary_df)
    else:
        st.info("No metadata available for this dataset.")


    st.markdown("### 🎨 Visualizations")

    # 1) Outcome distribution by treatment
    st.markdown("**Outcome distribution by treatment**")

    # Only plot if both selected columns exist and are numeric/categorical
    if treatment_col in df.columns and outcome_col in df.columns:
        chart_data = df[[treatment_col, outcome_col]].copy()
        chart_data = chart_data.rename(
            columns={treatment_col: "Treatment", outcome_col: "Outcome"}
        )

        # Try to ensure correct types
        try:
            chart_data["Treatment"] = chart_data["Treatment"].astype(str)
        except Exception:
            pass

        if chart_data["Treatment"].nunique() <= 20:
            hist = (
                alt.Chart(chart_data)
                .mark_bar(opacity=0.6)
                .encode(
                    alt.X("Outcome:Q", bin=alt.Bin(maxbins=30), title="Outcome"),
                    alt.Y("count()", stack=None, title="Count"),
                    color=alt.Color("Treatment:N", title="Treatment"),
                    tooltip=["Treatment", "count()"],
                )
                .properties(height=200)
            )
            st.altair_chart(hist, use_container_width=True)
        else:
            st.info("Too many treatment levels to visualize effectively.")
    else:
        st.warning("Please select valid Treatment and Outcome columns.")

    # 3) Confounder balance plot (optional)
    if len(confounders) > 0:
        st.markdown("**Confounder balance**")
        melt_list = []
        for c in confounders:
            grp = df.groupby(treatment_col)[c].mean().reset_index()
            grp.columns = ["Treatment", "MeanValue"]
            grp["Confounder"] = c
            melt_list.append(grp)
        bal_df = pd.concat(melt_list)
        conf_chart = (
            alt.Chart(bal_df)
            .mark_circle(size=90)
            .encode(
                x="MeanValue:Q",
                y=alt.Y("Confounder:N", title=None),
                color="Treatment:N",
                tooltip=["Confounder", "Treatment", "MeanValue"],
            )
            .properties(height=30 * len(confounders))
        )
        st.altair_chart(conf_chart, use_container_width=True)


# ===== RIGHT SIDE: Visualizations =====
with col2:
    # If repair run
    if st.session_state.run_repair:
        st.markdown("### 🧾 Repair summary")

        start_time = time.time()

        if algorithm.startswith("Tuple"):
            df_new, removed_df, new_ate = subcure_tuple(
                df,
                treatment_col,
                outcome_col,
                confounders,
                ate_target=desired_center,
                eps=tolerance,
            )
            algo_name = "Tuple-level repair"
        else:
            df_new, removed_df, new_ate = subcure_pattern(
                df,
                treatment_col,
                outcome_col,
                confounders,
                ate_target=desired_center,
                eps=tolerance,
            )
            algo_name = "Pattern-level repair"

        st.session_state.new_ate = new_ate

        exec_time = time.time() - start_time
        n_removed = len(removed_df)
        pct_removed = (n_removed / len(df)) * 100 if len(df) > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(f"{algo_name} runtime", f"{exec_time:.2f}s")
        with c2:
            st.metric("tuples removed", f"{n_removed} ({pct_removed:.2f}%)")
        with c3:
            st.metric("old ATE", f"{current_ate:.3f}")
        with c4:
            st.metric("new ATE", f"{new_ate:.3f}")

        # ===================== 🔍 ADDED INSIGHT SECTION =====================
        st.markdown("---")
        st.subheader("📊 Analysis of Removed Data (Influential Tuples)")

        if not removed_df.empty:
            st.markdown("#### Removed Tuples")
            st.dataframe(
                removed_df.head(10),
                use_container_width=True,
                height=200
            )

            retained_df = df_new.copy()

            # ================= Numeric feature histogram =================
            numeric_cols = removed_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                feature = numeric_cols[0]
                st.markdown(f"**Distribution of numeric feature:** `{feature}`")

                merged_plot_df = pd.concat([
                    pd.DataFrame({'Category': 'Removed', 'Value': removed_df[feature]}),
                    pd.DataFrame({'Category': 'Retained', 'Value': retained_df[feature]})
                ])

                chart_hist = (
                    alt.Chart(merged_plot_df)
                    .mark_bar(opacity=0.7)
                    .encode(
                        x=alt.X('Value:Q', bin=alt.Bin(maxbins=25), title=feature),
                        y=alt.Y('count()', stack=None, title='Count'),
                        color=alt.Color('Category:N', scale=alt.Scale(range=['#FF8C00', '#1f77b4'])),
                        tooltip=['Category', 'count()'],
                    )
                    .properties(height=200)
                )
                st.altair_chart(chart_hist, use_container_width=True)

            # ================= Categorical feature comparison =================
            cat_cols = removed_df.select_dtypes(exclude=[np.number]).columns.tolist()
            if len(cat_cols) > 0:
                feature_cat = cat_cols[0]
                st.markdown(f"**Distribution of categorical feature:** `{feature_cat}`")

                cat_removed = removed_df[feature_cat].value_counts().reset_index()
                cat_removed.columns = ['Category', 'Count']
                cat_removed['Set'] = 'Removed'

                cat_retained = retained_df[feature_cat].value_counts().reset_index()
                cat_retained.columns = ['Category', 'Count']
                cat_retained['Set'] = 'Retained'

                cat_df = pd.concat([cat_removed, cat_retained])

                chart_cat = (
                    alt.Chart(cat_df)
                    .mark_bar()
                    .encode(
                        x=alt.X('Count:Q', title='Record Count'),
                        y=alt.Y('Category:N', title=feature_cat),
                        color=alt.Color(
                            'Set:N',
                            scale=alt.Scale(
                                domain=['Removed', 'Retained'],
                                range=['#FF8C00', '#1f77b4']
                            ),
                        ),
                        tooltip=['Set', 'Category', 'Count'],
                    )
                    .properties(height=250)
                )
                st.altair_chart(chart_cat, use_container_width=True)

        else:
            st.warning("No removed tuples available for visualization.")

