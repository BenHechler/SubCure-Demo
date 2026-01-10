import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from Datasets import load_ACS_dataset, load_twins_dataset, load_german_dataset, load_SO_dataset
from Algorithms import subcure_tuple, subcure_pattern, estimate_ate_linear

import time
st.set_page_config(page_title="SubCure Demo", layout="wide")

# ---------------------- SESSION STATE DEFAULTS ----------------------
defaults = {
    "df": None,
    "ate_val": None,
    "desired_center": None,
    "tolerance": 10,
    "algorithm": "Tuple-level (remove individual rows)",
    "max_removals": 20,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------- SIDEBAR CONFIG ----------------------
if "new_ate" not in st.session_state:
    st.session_state.new_ate = None

if "ate_computed" not in st.session_state:
    st.session_state.ate_computed = False


st.sidebar.title("⚙️ Controls")

dataset_name = st.sidebar.selectbox(
    "Dataset",
    ["ACS", "Credit", "Stack Overflow", "Twins", "Upload CSV"]
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

if st.sidebar.button("Compute causal effect"):
    st.session_state.ate_val = estimate_ate_linear(df, treatment_col, outcome_col, confounders)
    st.session_state.ate_computed = True

if st.session_state.ate_val is not None:
    st.sidebar.markdown(
        f"**Current causal effect:** `{st.session_state.ate_val:.1f}`"
    )
else:
    st.sidebar.info("Click **Compute ATE** to calculate it.")


current_ate = estimate_ate_linear(df, treatment_col, outcome_col, confounders)
desired_center = st.sidebar.number_input(
    "Desired causal effect",
    value=float(np.round(current_ate, 3)),
    step=1.0
)

tolerance_percanteges = st.sidebar.slider(
    "Tolerance in % (±)",
    min_value=0,
    max_value=50,
    value=st.session_state.tolerance,
    step=1,
    help="Allowed deviation around the target ATE"
)

tolerance = desired_center * tolerance_percanteges / 100

algorithm = st.sidebar.selectbox(
    "Intervention algorithm",
    ["Tuple-level (remove individual rows)", "Pattern-level (remove subpopulations)"]
)

if "run_repair" not in st.session_state:
    st.session_state.run_repair = False

if st.sidebar.button("Run algorithm"):
    st.session_state.run_repair = True

if st.session_state.new_ate is not None:
    st.sidebar.markdown(
        f"**New causal effect after repair:** `{st.session_state.new_ate:.1f}`"
    )


# ---------------------- MAIN PANEL ----------------------
col_logo, col_title = st.columns([0.2, 4])

with col_logo:
    st.image("subcure_logo.png", width=120)

with col_title:
    st.title("SubCure Demo – Stress Testing Causal Claims")


col1, space, col2 = st.columns([1, 0.05, 1])


acs_message = f"""##### Current state: people without a disability earn {round(current_ate, 1)} more than people with disability on average"""
credit_message = f"""##### Current state: people without a house has {round(current_ate, 1)} higher credit risk than people without a house on average"""
so_message = f"""##### Current state: people without a formal education earn {abs(round(current_ate, 1))} less than people with a formal education on average"""
twins_message = f"""##### Current state: people without a heavier twin has {abs(round(current_ate, 1))} less mortality than people with heavier twin on average"""
# twins_message = "to do 2"
goal_dict = {"ACS": acs_message, "Credit": credit_message, "Twins": twins_message, "Stack Overflow": so_message}
message2 = f"""##### Goal: change the causal effect from {round(current_ate, 1)} to {round(desired_center,1)}"""


# ===== LEFT SIDE: Dataset and Results =====
with col1:
    if current_ate and desired_center:
        if dataset_name != "Upload CSV":
            goal_m = goal_dict.get(dataset_name)
            if st.session_state.ate_computed:
                st.markdown(goal_m)
                if round(current_ate, 1) != round(desired_center, 1):
                    st.markdown(message2)

    st.markdown("### 📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True, height=180)

    num_records = len(df)
    num_atts = len(df.columns)
    st.markdown(
        f"**Records:** {num_records:,} | "
        f"**Attributes:** {num_atts}"
    )

    st.markdown("#### Outcome distribution by treatment")

    # Only plot if both selected columns exist and are numeric/categorical
    if treatment_col in df.columns and outcome_col in df.columns:
        chart_data = df[[treatment_col, outcome_col]].copy()
        chart_data = chart_data.rename(
            columns={treatment_col: "Treatment", outcome_col: "Outcome"}
        )

        try:
            chart_data["Treatment"] = chart_data["Treatment"].astype(str).apply(lambda x: f"{treatment_col}: {x}")
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

    # ai_text = ra.ai_dataset_one_liner(df, treatment_col, outcome_col)
    # st.write(ai_text)

    # If repair run
    if st.session_state.run_repair:
        st.markdown("### 🧾 Results Summary")

        start_time = time.time()

        if algorithm.startswith("Tuple"):
            df_new, removed_df_raw, new_ate = subcure_tuple(
                df,
                treatment_col,
                outcome_col,
                confounders,
                ate_target=desired_center,
                eps=tolerance,
            )
            algo_name = "Tuple-level repair"
        else:
            df_new, removed_df_raw, new_ate = subcure_pattern(
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
        n_removed = len(removed_df_raw)
        pct_removed = (n_removed / len(df)) * 100 if len(df) > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(f"runtime", f"{exec_time:.1f}s")
        with c2:
            st.metric("tuples removed", f"{pct_removed:.1f}%")
        with c3:
            st.metric("old causal effect", f"{current_ate:.1f}")
        with c4:
            if desired_center - tolerance <= new_ate <= desired_center + tolerance:
                st.metric("new causal effect", f"{new_ate:.1f}", "inside the desired range")
            else:
                st.metric("new causal effect", f"{new_ate:.1f}", "outside the desired range", delta_color="inverse")

    # ===== RIGHT SIDE: Visualizations =====
    with col2:
        if st.session_state.run_repair:
            st.markdown("### 📉 Averages Percentage Change (Caused By Removals)")
            # Extract numeric columns that exist in both datasets
            removed_df = df.loc[df.index.difference(df_new.index)].copy()

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c in removed_df.columns]

            comparison_rows = []
            for col in numeric_cols:
                orig = df[col].mean()
                rem = removed_df[col].mean()
                if orig == 0:
                    pct_diff = np.nan
                else:
                    pct_diff = ((rem - orig) / abs(orig)) * 100

                comparison_rows.append({
                    "Feature": col,
                    "Original_Mean": orig,
                    "Removed_Mean": rem,
                    "Diff_Percent": pct_diff
                })

            mean_df = pd.DataFrame(comparison_rows)
            mean_df["AbsPctDiff"] = mean_df["Diff_Percent"]
            plot_df = mean_df.copy()

            max_val = max(plot_df["AbsPctDiff"].abs())
            if max_val > 0:
                axis_upper_bound = round(1.5 * max_val)
            else:
                axis_upper_bound = 100

            chart = (
                alt.Chart(plot_df)
                .mark_bar()
                .encode(
                    x=alt.X("Feature:N",
                            sort=plot_df["AbsPctDiff"].sort_values(ascending=False).index.tolist(),
                            axis=alt.Axis(labelFontSize=13.5, titleFontSize=16, labelAngle=-45)),
                    y=alt.Y("AbsPctDiff:Q", title="Percentage Change (%)", scale=alt.Scale(domain=[-axis_upper_bound, axis_upper_bound])),
                    tooltip=[
                        alt.Tooltip("Feature:N"),
                        alt.Tooltip("Original_Mean:Q", format="1", title="Original Mean"),
                        alt.Tooltip("Removed_Mean:Q", format=".1f", title="Removed Mean"),
                        alt.Tooltip("Diff_Percent:Q", format=".1f", title="% Change"),
                    ]
                )
                .properties(
                    width=40 * len(plot_df),
                    height=550,
                    # padding={"bottom": 1}
                )
            )

            st.altair_chart(chart, use_container_width=True)

            insight_df = plot_df.copy()
            insight_df["abs_diff"] = insight_df["Diff_Percent"].abs()
            insight_df = insight_df.sort_values(by="abs_diff", ascending=False).reset_index(drop=True)
            # st.dataframe(insight_df)

            top_changes = f"""
            """

            for i in range(5):
                feature_name = insight_df.loc[i, "Feature"]
                feature_change = round(insight_df.loc[i, "Diff_Percent"], 1)
                increase_or_decrease = "increased by" if feature_change > 0 else " decreased by"
                abs_feature_change = abs(feature_change)
                old = round(insight_df.loc[i, "Original_Mean"], 1)
                new = round(insight_df.loc[i, "Removed_Mean"], 1)

                added_text = f"""
    {i+1}. {feature_name} {increase_or_decrease} {abs_feature_change}% (from {old} to {new})
    """
                top_changes += added_text

            if max_val > 0:
                st.markdown("### 🧠 Insights on Removed Subpopulation")
                st.markdown("##### Top 5 most significant feature shifts:")
                st.write(top_changes)
            else:
                st.markdown("### No records has been removed!")
