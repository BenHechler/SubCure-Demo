import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from Datasets import load_ACS_dataset, load_twins_dataset, load_german_dataset, load_SO_dataset
from Algorithms import subcure_tuple, subcure_pattern, estimate_ate_linear
import removed_analysis as ra

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

if st.session_state.ate_val is not None:
    st.sidebar.markdown(
        f"**Current causal effect:** `{st.session_state.ate_val:.1f}`"
    )
else:
    st.sidebar.info("Click **Compute ATE** to calculate it.")


current_ate = estimate_ate_linear(df, treatment_col, outcome_col, confounders)
desired_center = st.sidebar.number_input(
    "Desired causal effect",
    value=float(np.round(current_ate, 3))
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
    st.image("subcure_logo_2.png", width=120)

with col_title:
    st.title("SubCure Demo – Stress Testing Causal Claims")


col1, space, col2 = st.columns([1, 0.05, 1])

# ===== LEFT SIDE: Dataset and Results =====
with col1:
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
        # mean_df["Direction"] = mean_df["Diff_Percent"].apply(lambda x: "Positive" if x >= 0 else "Negative")

        plot_df = mean_df.copy()

        # Color scale red for negative, green for positive
        # color_scale = alt.Scale(
        #     domain=["Positive", "Negative"],
        #     range=["#4daf4a", "#e41a1c"]  # green, red
        # )
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
                        title="Feature"),
                y=alt.Y("AbsPctDiff:Q", title="Percentage Change (%)", scale=alt.Scale(domain=[-axis_upper_bound, axis_upper_bound])),
                # color=alt.Color("Direction:N", scale=color_scale, legend=alt.Legend(title="Direction")),
                tooltip=[
                    alt.Tooltip("Feature:N"),
                    alt.Tooltip("Original_Mean:Q", format="1", title="Original Mean"),
                    alt.Tooltip("Removed_Mean:Q", format=".1f", title="Removed Mean"),
                    alt.Tooltip("Diff_Percent:Q", format=".1f", title="% Change"),
                ]
            )
            .properties(
                width=40 * len(plot_df),
                height=450,
                # padding={"bottom": 1}
            )
        )

        st.altair_chart(chart, use_container_width=True)

        if removed_df is None or removed_df.empty:
            st.info("No removed tuples to analyze yet.")
        else:
            results, diff = ra.compute_removed_analysis(removed_df, df)

        st.markdown("## 🤖 Insights on Removed Subpopulation")

        with st.spinner("Generating interpretability insights..."):
            # try:
            #     ai_text = ra.call_gpt_for_removed_analysis(results, diff)
            # except Exception as e:
            #     ai_text = f"⚠️ GPT call failed: {e}"
            ai_text = f"⚠️ GPT call blocked - saving tokens for now"

        st.write(ai_text)
