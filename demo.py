import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from Datasets import load_ACS_dataset, load_twins_dataset, load_german_dataset, load_SO_dataset, DATASET_META
from Algorithms import subcure_tuple, subcure_pattern, estimate_ate_linear

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

# Column selection
all_cols = list(df.columns)
treatment_col = st.sidebar.selectbox("Treatment (binary 0/1)", all_cols)
outcome_col = st.sidebar.selectbox("Outcome (numeric)", all_cols)
confounders = st.sidebar.multiselect(
    "Confounders (optional)",
    [c for c in all_cols if c not in [treatment_col, outcome_col]],
    default=[]
)

# Compute ATE button + dynamic display
if "ate_val" not in st.session_state:
    st.session_state.ate_val = None

if st.sidebar.button("Compute ATE"):
    st.session_state.ate_val = estimate_ate_linear(df, treatment_col, outcome_col, confounders)

# show result below button, before desired ATE input
if st.session_state.ate_val is not None:
    st.sidebar.markdown(
        f"**Current ATE:** `{st.session_state.ate_val:.4f}`"
    )
else:
    st.sidebar.info("Click **Compute ATE** to calculate it.")


# Target ATE range
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


run_repair = st.sidebar.button("Run repair")

# ---------------------- MAIN PANEL ----------------------
st.title("🧠 SubCure Demo – Stress Testing Causal Claims")

col1, col2 = st.columns([1.4, 1])

# ===== LEFT SIDE: Dataset and Results =====
with col1:
    st.markdown("### 📊 Dataset preview")
    st.dataframe(df.head(), use_container_width=True, height=180)

    st.markdown("### 📘 Dataset summary (from paper Table 1)")
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

    # st.markdown("### 🧮 ATE computation")
    # if st.session_state.ate_val is not None:
    #     st.success(f"Current ATE = **{st.session_state.ate_val:.4f}**")
    # else:
    #     st.info("Click **Compute ATE** in the sidebar to calculate the current ATE.")

    # If repair run
    if run_repair:
        if algorithm.startswith("Tuple"):
            df_new, removed_df, new_ate = subcure_tuple(
                df,
                treatment_col,
                outcome_col,
                confounders,
                ate_target=desired_center,
                eps=tolerance,
                max_iters=1000,
            )
            st.success(f"New ATE after tuple-level repair: **{new_ate:.4f}**")
            st.write(f"Total tuples removed: **{len(removed_df)}**")
            st.markdown("#### Removed tuples")
            st.dataframe(removed_df.head(10), use_container_width=True, height=180)
        else:
            df_new, removed_df, new_ate = subcure_pattern(
                df,
                treatment_col,
                outcome_col,
                confounders,
                ate_target=desired_center,
                eps=tolerance,
                max_walks=1000,
            )
            st.success(f"New ATE after pattern-level repair: **{new_ate:.4f}**")
            st.write(f"Total patterns removed: **{len(removed_df)}**")
            st.markdown("#### Removed subpopulations")
            st.dataframe(removed_df.head(10), use_container_width=True, height=180)

        st.caption(
            "Tuple-level removes individual rows, while Pattern-level removes attribute=value groups."
        )

    # ===================== 🔍 ADDED INSIGHT SECTION =====================
        st.markdown("---")
        st.subheader("📊 Analysis of Removed Data (Influential Tuples)")

        if not removed_df.empty:
            retained_df = df_new.copy()
            try:
                # Outcome comparison
                removed_rate = removed_df[outcome_col].mean()
                retained_rate = retained_df[outcome_col].mean()
                diff_rate = removed_rate - retained_rate

                st.metric(
                    label=f"Mean Outcome (Removed vs Retained)",
                    value=f"{removed_rate:.3f}",
                    delta=f"{diff_rate:+.3f}",
                    delta_color="inverse" if diff_rate > 0 else "normal"
                )
            except Exception:
                st.warning("Outcome column not numeric; skipping mean comparison.")

            # Birth weight–like continuous variable detection
            numeric_cols = removed_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                feature = numeric_cols[0]
                st.markdown(f"**Distribution of key numeric feature:** `{feature}`")

                merged_plot_df = pd.concat([
                    pd.DataFrame({'Category': 'Removed', 'Value': removed_df[feature]}),
                    pd.DataFrame({'Category': 'Retained', 'Value': retained_df[feature]})
                ])

                chart_hist = alt.Chart(merged_plot_df).mark_bar(opacity=0.7).encode(
                    x=alt.X('Value:Q', bin=alt.Bin(maxbins=25), title=feature),
                    y=alt.Y('count()', stack=None, title='Count'),
                    color=alt.Color('Category:N', scale=alt.Scale(range=['#FF8C00', '#1f77b4'])),
                    tooltip=['Category', 'count()']
                ).properties(height=200)
                st.altair_chart(chart_hist, use_container_width=True)

            # Categorical variable comparison (if any)
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

                chart_cat = alt.Chart(cat_df).mark_bar().encode(
                    x=alt.X('Count:Q', title='Record Count'),
                    y=alt.Y('Category:N', title=feature_cat),
                    color=alt.Color('Set:N', scale=alt.Scale(domain=['Removed', 'Retained'],
                                                             range=['#FF8C00', '#1f77b4'])),
                    tooltip=['Set', 'Category', 'Count']
                ).properties(height=250)
                st.altair_chart(chart_cat, use_container_width=True)

            st.info(
                "Removed tuples often correspond to influential, low-frequency or low-risk groups "
                "that artificially strengthened the original causal estimate. Visualizing them helps "
                "understand where model fragility originates."
            )
        else:
            st.warning("No removed tuples available for visualization.")
        # ====================================================================

# ===== RIGHT SIDE: Visualizations =====
with col2:
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


    # 2) ATE vs desired visualization
    st.markdown("**ATE vs desired range**")
    left_bound = min(current_ate, desired_center - tolerance) - abs(tolerance) * 0.6
    right_bound = max(current_ate, desired_center + tolerance) + abs(tolerance) * 0.6
    band_df = pd.DataFrame({
        "start": [desired_center - tolerance],
        "end": [desired_center + tolerance],
    })
    band = alt.Chart(band_df).mark_rect(opacity=0.3, color="green").encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[left_bound, right_bound])),
        x2="end:Q"
    )
    current_line = alt.Chart(pd.DataFrame({"current": [current_ate]})).mark_rule(
        color="red", strokeWidth=3
    ).encode(x="current:Q")
    center_line = alt.Chart(pd.DataFrame({"center": [desired_center]})).mark_rule(
        color="green", strokeDash=[4, 4]
    ).encode(x="center:Q")
    labels = alt.Chart(
        pd.DataFrame(
            {"label": ["Current ATE", "Target center"], "value": [current_ate, desired_center]}
        )
    ).mark_text(align="left", dx=4, dy=-6, fontSize=11).encode(x="value:Q", text="label:N")
    ate_chart = (band + current_line + center_line + labels).properties(height=120)
    st.altair_chart(ate_chart, use_container_width=True)

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
