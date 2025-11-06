import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="SubCure Demo", layout="wide")

st.title("SubCure Demo")

st.markdown(
    """
A minimal demo for **“Stress-Testing Causal Claims via Cardinality Repairs”**.

**Flow**
1. Choose dataset  
2. Select treatment, outcome, confounders  
3. Compute ATE  
4. Set target ATE range  
5. Choose repair algorithm (tuple-level or pattern-level)  
6. Run the repair and inspect removed data
"""
)

# ---------------------------------------------------------------------
# Load example datasets
# ---------------------------------------------------------------------
@st.cache_data
def load_twins_dataset():
    np.random.seed(42)
    n = 500
    treatment = np.random.binomial(1, 0.7, size=n)
    edu_years = np.random.randint(8, 20, size=n)
    age = np.random.randint(20, 65, size=n)
    base = 15000 + 1200 * edu_years + 200 * (age - 30)
    treatment_effect = 6000
    noise = np.random.normal(0, 4000, size=n)
    wage = base + treatment * treatment_effect + noise
    return pd.DataFrame({
        "no_disability": treatment,
        "wage": wage,
        "edu_years": edu_years,
        "age": age,
    })


@st.cache_data
def load_ACS_dataset():
    np.random.seed(123)
    n = 400
    got_treatment = np.random.binomial(1, 0.5, size=n)
    severity = np.random.randint(1, 5, size=n)
    age = np.random.randint(18, 80, size=n)
    base = 20 + 3 * severity + 0.1 * (age - 40)
    outcome = base - 2 * got_treatment + np.random.normal(0, 1.5, size=n)
    return pd.DataFrame({
        "got_treatment": got_treatment,
        "recovery_days": outcome,
        "severity": severity,
        "age": age,
    })


dataset_name = st.sidebar.selectbox("Choose dataset", ["Twins", "ACS", "Upload CSV"])
if dataset_name == "Twins":
    df = load_twins_dataset()
elif dataset_name == "ACS":
    df = load_ACS_dataset()
else:
    uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if uploaded is None:
        st.stop()
    df = pd.read_csv(uploaded)

st.subheader("Dataset preview")
st.dataframe(df.head())

# ---------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------
all_cols = list(df.columns)
treatment_col = st.selectbox("Treatment (binary 0/1)", all_cols)
outcome_col = st.selectbox("Outcome (numeric)", all_cols, index=min(1, len(all_cols) - 1))
confounders = st.multiselect(
    "Confounders (optional)",
    [c for c in all_cols if c not in [treatment_col, outcome_col]],
    default=[],
)

def compute_ate_linear(data, treat, outcome, confs):
    X = data[[treat] + confs].values
    y = data[outcome].values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]
    if len(y) == 0:
        return np.nan
    model = LinearRegression().fit(X, y)
    return model.coef_[0]

if st.button("Compute ATE"):
    ate_val = compute_ate_linear(df, treatment_col, outcome_col, confounders)
    st.success(f"Current ATE = **{ate_val:.4f}**")

# ---------------------------------------------------------------------
# Target range and algorithm
# ---------------------------------------------------------------------
current_ate = compute_ate_linear(df, treatment_col, outcome_col, confounders)
desired_center = st.number_input("Desired ATE center", value=float(np.round(current_ate, 3)))
tolerance = st.number_input("Tolerance (±)", value=0.5, min_value=0.0, step=0.1)
algorithm = st.selectbox(
    "Choose repair algorithm",
    ["Tuple-level (remove individual rows)", "Pattern-level (remove subpopulations)"],
)
max_removals = st.slider("Max iterations", 1, 100, 20)

def in_range(v, c, tol):
    return (c - tol) <= v <= (c + tol)

# ---------------------------------------------------------------------
# Tuple-level algorithm
# ---------------------------------------------------------------------
def run_tuple_repair(df_in):
    df_work = df_in.copy()
    removed_rows = []
    current = compute_ate_linear(df_work, treatment_col, outcome_col, confounders)
    for _ in range(max_removals):
        if in_range(current, desired_center, tolerance):
            break
        best_idx, best_ate, best_dist = None, None, None
        for idx in df_work.index:
            tmp = df_work.drop(index=idx)
            ate_tmp = compute_ate_linear(tmp, treatment_col, outcome_col, confounders)
            dist = abs(ate_tmp - desired_center)
            if best_dist is None or dist < best_dist:
                best_idx, best_ate, best_dist = idx, ate_tmp, dist
        if best_idx is None:
            break
        removed_rows.append(df_work.loc[best_idx])
        df_work = df_work.drop(index=best_idx)
        current = best_ate
    return df_work, pd.DataFrame(removed_rows), current

# ---------------------------------------------------------------------
# Pattern-level algorithm
# ---------------------------------------------------------------------
def run_pattern_repair(df_in):
    df_work = df_in.copy()
    removed_patterns = []
    current = compute_ate_linear(df_work, treatment_col, outcome_col, confounders)
    other_cols = [c for c in df_work.columns if c not in [treatment_col, outcome_col]]
    for _ in range(max_removals):
        if in_range(current, desired_center, tolerance):
            break
        best_col = best_val = best_ate = None
        best_dist = None
        for col in other_cols:
            for val in df_work[col].unique()[:30]:
                tmp = df_work[df_work[col] != val]
                ate_tmp = compute_ate_linear(tmp, treatment_col, outcome_col, confounders)
                dist = abs(ate_tmp - desired_center)
                if best_dist is None or dist < best_dist:
                    best_col, best_val, best_ate, best_dist = col, val, ate_tmp, dist
        if best_col is None:
            break
        removed_count = (df_work[best_col] == best_val).sum()
        removed_patterns.append((best_col, best_val, removed_count))
        df_work = df_work[df_work[best_col] != best_val]
        current = best_ate
    removed_df = pd.DataFrame(removed_patterns, columns=["Column", "Value", "Rows_removed"])
    return df_work, removed_df, current

# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if st.button("Run repair"):
    if algorithm.startswith("Tuple"):
        df_new, removed_df, new_ate = run_tuple_repair(df)
        st.success(f"New ATE after tuple-level repair: **{new_ate:.4f}**")
        st.write(f"Total tuples removed: **{len(removed_df)}**")
        st.subheader("Removed tuples")
        st.dataframe(removed_df.head())
    else:
        df_new, removed_df, new_ate = run_pattern_repair(df)
        st.success(f"New ATE after pattern-level repair: **{new_ate:.4f}**")
        st.write(f"Total patterns removed: **{len(removed_df)}**")
        st.subheader("Head of removed subpopulations")
        st.dataframe(removed_df.head())

    st.info(
        "Note: tuple-level removes rows individually. "
        "Pattern-level removes attribute=value groups (simplified)."
    )
