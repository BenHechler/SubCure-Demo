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
    ["ACS", "Credit", "Stack Overflow", "Upload CSV"]  # Twins dataset has no csv
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

    num_records = len(df)
    num_atts = len(df.columns)
    st.markdown(
        f"**Records:** {num_records:,} | "
        f"**Attributes:** {num_atts}"
    )

    st.markdown("### 🎨 Visualizations")

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






        # =================== Attribute Pattern Discovery ===================
        st.markdown("#### 🧬 Common Attribute Patterns Among Removed Tuples")

        categorical_cols = removed_df.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_cols = removed_df.select_dtypes(include=[np.number]).columns.tolist()
        results_cat, results_num = [], []

        # --- Categorical features: frequency lift ---
        for col in categorical_cols:
            top_removed = removed_df[col].value_counts(normalize=True).head(1)
            if top_removed.empty:
                continue
            top_val = top_removed.index[0]
            removed_pct = top_removed.iloc[0] * 100
            total_pct = df[col].value_counts(normalize=True).reindex([top_val]).fillna(0).iloc[0] * 100
            lift = removed_pct / (total_pct + 1e-6)
            results_cat.append({
                "Attribute": col,
                "Top Value": str(top_val),
                "Removed %": removed_pct,
                "Overall %": total_pct,
                "Overrepresentation (×)": lift
            })

        # --- Numeric features: mean shift in SD units ---
        for col in numeric_cols:
            rmv_mean, all_mean = removed_df[col].mean(), df[col].mean()
            std = np.std(df[col]) if np.std(df[col]) > 0 else 1e-6
            diff_std = (rmv_mean - all_mean) / std
            results_num.append({
                "Attribute": col,
                "Top Value": "Higher" if diff_std > 0 else "Lower",
                "Removed %": np.nan,
                "Overall %": np.nan,
                "Overrepresentation (×)": abs(diff_std)
            })

        df_cat = pd.DataFrame(results_cat)
        df_num = pd.DataFrame(results_num)

        if df_cat.empty and df_num.empty:
            st.info("No distinctive attributes found among removed tuples.")
        else:
            st.markdown("#### 📊 Top Distinctive Attributes")
            if not df_cat.empty:
                st.markdown("**Categorical features (frequency lift):**")
                df_cat = df_cat.sort_values("Overrepresentation (×)", ascending=False)
                st.dataframe(df_cat.head(10), use_container_width=True, height=250)
            if not df_num.empty:
                st.markdown("**Numeric features (standard deviation shift):**")
                df_num = df_num.sort_values("Overrepresentation (×)", ascending=False)
                st.dataframe(df_num.head(10), use_container_width=True, height=250)

            # --- Add color classification ---
            def assign_color(score):
                if score > 2:
                    return "Strong"
                elif score > 1.3:
                    return "Moderate"
                else:
                    return "Weak"


            for df_temp in [df_cat, df_num]:
                if not df_temp.empty:
                    df_temp["color_code"] = df_temp["Overrepresentation (×)"].apply(assign_color)

            color_scale = alt.Scale(
                domain=["Strong", "Moderate", "Weak"],
                range=["#e41a1c", "#ffbf00", "#4daf4a"]
            )

            # --- Side-by-side bar charts ---
            charts = []
            if not df_cat.empty:
                chart_cat = (
                    alt.Chart(df_cat.head(15))
                    .mark_bar()
                    .encode(
                        x=alt.X("Overrepresentation (×):Q", title="Lift (Removed vs Overall)"),
                        y=alt.Y("Attribute:N", sort="-x"),
                        color=alt.Color("color_code:N", scale=color_scale, legend=None),
                        tooltip=["Attribute", "Top Value",
                                 alt.Tooltip("Removed %:Q", format=".1f"),
                                 alt.Tooltip("Overall %:Q", format=".1f"),
                                 alt.Tooltip("Overrepresentation (×):Q", format=".2f")]
                    )
                    .properties(title="Categorical Attributes", height=25 * min(len(df_cat), 15))
                )
                charts.append(chart_cat)

            if not df_num.empty:
                chart_num = (
                    alt.Chart(df_num.head(15))
                    .mark_bar()
                    .encode(
                        x=alt.X("Overrepresentation (×):Q", title="Mean Shift (|Δ| SD)"),
                        y=alt.Y("Attribute:N", sort="-x"),
                        color=alt.Color("color_code:N", scale=color_scale,
                                        legend=alt.Legend(title="Influence Strength")),
                        tooltip=["Attribute", "Top Value",
                                 alt.Tooltip("Overrepresentation (×):Q", format=".2f")]
                    )
                    .properties(title="Numeric Attributes", height=25 * min(len(df_num), 15))
                )
                charts.append(chart_num)

            if charts:
                st.altair_chart(alt.hconcat(*charts), use_container_width=True)

            # --- 🧠 Color-coded automatic summary generation ---
            combined_df = pd.concat([df_cat, df_num], ignore_index=True)
            top3 = combined_df.sort_values("Overrepresentation (×)", ascending=False).head(3)
            insights = []
            for _, row in top3.iterrows():
                attr, val, score = row["Attribute"], row["Top Value"], row["Overrepresentation (×)"]
                color_emoji = "🟩"
                if score > 2:
                    color_emoji = "🟥"
                elif score > 1.3:
                    color_emoji = "🟨"

                if not np.isnan(row.get("Removed %", np.nan)):
                    pct = f"{row['Removed %']:.1f}%"
                    insights.append(f"{color_emoji} {attr} = **{val}** ({pct} of removed, ×{score:.1f})")
                else:
                    insights.append(f"{color_emoji} {attr} tends to be **{val}** (|Δ| ≈ {score:.1f} SD)")

            summary_text = ", ".join(insights)
            st.markdown(f"#### 🗣️ Summary\nRemoved tuples are predominantly characterized by: {summary_text}.")
            st.caption(
                "🟥 Strongly overrepresented | 🟨 Moderately distinctive | 🟩 Minor difference\n"
                "Left: categorical attributes showing frequency lift; Right: numeric features showing mean shift.\n"
                "Together they highlight which traits most define the tuples whose removal influenced the causal estimate."
            )
        #
        # # ===================== 🔍 ADDED INSIGHT SECTION =====================
        # st.markdown("---")
        # st.subheader("📊 Analysis of Removed Data (Influential Tuples)")
        #
        # if not removed_df.empty:
        #     st.markdown("#### Removed Tuples")
        #     st.dataframe(
        #         removed_df.head(10),
        #         use_container_width=True,
        #         height=200
        #     )
        #
        #     retained_df = df_new.copy()
        #
        #     # ================= Numeric feature histogram =================
        #     numeric_cols = removed_df.select_dtypes(include=[np.number]).columns.tolist()
        #     if len(numeric_cols) > 0:
        #         feature = numeric_cols[0]
        #         st.markdown(f"**Distribution of numeric feature:** `{feature}`")
        #
        #         merged_plot_df = pd.concat([
        #             pd.DataFrame({'Category': 'Removed', 'Value': removed_df[feature]}),
        #             pd.DataFrame({'Category': 'Retained', 'Value': retained_df[feature]})
        #         ])
        #
        #         chart_hist = (
        #             alt.Chart(merged_plot_df)
        #             .mark_bar(opacity=0.7)
        #             .encode(
        #                 x=alt.X('Value:Q', bin=alt.Bin(maxbins=25), title=feature),
        #                 y=alt.Y('count()', stack=None, title='Count'),
        #                 color=alt.Color('Category:N', scale=alt.Scale(range=['#FF8C00', '#1f77b4'])),
        #                 tooltip=['Category', 'count()'],
        #             )
        #             .properties(height=200)
        #         )
        #         st.altair_chart(chart_hist, use_container_width=True)
        #
        #     # ================= Categorical feature comparison =================
        #     cat_cols = removed_df.select_dtypes(exclude=[np.number]).columns.tolist()
        #     if len(cat_cols) > 0:
        #         feature_cat = cat_cols[0]
        #         st.markdown(f"**Distribution of categorical feature:** `{feature_cat}`")
        #
        #         cat_removed = removed_df[feature_cat].value_counts().reset_index()
        #         cat_removed.columns = ['Category', 'Count']
        #         cat_removed['Set'] = 'Removed'
        #
        #         cat_retained = retained_df[feature_cat].value_counts().reset_index()
        #         cat_retained.columns = ['Category', 'Count']
        #         cat_retained['Set'] = 'Retained'
        #
        #         cat_df = pd.concat([cat_removed, cat_retained])
        #
        #         chart_cat = (
        #             alt.Chart(cat_df)
        #             .mark_bar()
        #             .encode(
        #                 x=alt.X('Count:Q', title='Record Count'),
        #                 y=alt.Y('Category:N', title=feature_cat),
        #                 color=alt.Color(
        #                     'Set:N',
        #                     scale=alt.Scale(
        #                         domain=['Removed', 'Retained'],
        #                         range=['#FF8C00', '#1f77b4']
        #                     ),
        #                 ),
        #                 tooltip=['Set', 'Category', 'Count'],
        #             )
        #             .properties(height=250)
        #         )
        #         st.altair_chart(chart_cat, use_container_width=True)
        #
        # else:
        #     st.warning("No removed tuples available for visualization.")

