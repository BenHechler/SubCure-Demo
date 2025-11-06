import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="SubCure Demo", layout="wide")

st.title("SubCure Demo")  # - Causal ATE  - Cardinality Repair

st.markdown(
    """
This is a demo for the **“Stress-Testing Causal Claims via Cardinality Repairs”** paper.

**Flow**:
1. Choose dataset  
2. Pick treatment / outcome / confounders  
3. Compute ATE  
4. Set desired ATE range  
5. **Choose algorithm** (tuple-level vs pattern-level)  
6. Run the chosen algorithm to push ATE into the desired range
"""
)

# ------------------------------------------------------------
# 1) DATASETS
# ------------------------------------------------------------
@st.cache_data
def load_twins_dataset():
    # synthetic dataset with binary disability and wages
    np.random.seed(42)
    n = 500
    treatment = np.random.binomial(1, 0.7, size=n)
    edu_years = np.random.randint(8, 20, size=n)
    age = np.random.randint(20, 65, size=n)
    base = 15000 + 1200 * edu_years + 200 * (age - 30)
    treatment_effect = 6000
    noise = np.random.normal(0, 4000, size=n)
    wage = base + treatment * treatment_effect + noise
    df = pd.DataFrame(
        {
            "no_disability": treatment,
            "wage": wage,
            "edu_years": edu_years,
            "age": age,
        }
    )
    return df


@st.cache_data
def load_ACS_dataset():
    np.random.seed(123)
    n = 400
    got_treatment = np.random.binomial(1, 0.5, size=n)
    severity = np.random.randint(1, 5, size=n)
    age = np.random.randint(18, 80, size=n)
    base = 20 + 3 * severity + 0.1 * (age - 40)
    outcome = base - 2 * got_treatment + np.random.normal(0, 1.5, size=n)
    df = pd.DataFrame(
        {
            "got_treatment": got_treatment,
            "recovery_days": outcome,
            "severity": severity,
            "age": age,
        }
    )
    return df


dataset_name = st.sidebar.selectbox(
    "Choose a dataset",
    ["Twins", "ACS", "Upload CSV"],
)

if dataset_name == "Twins":
    df = load_twins_dataset()
elif dataset_name == "ACS":
    df = load_ACS_dataset()
else:
    uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.warning("Please upload a CSV to continue.")
        st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head())

# ------------------------------------------------------------
# 2) COLUMN SELECTION
# ------------------------------------------------------------
st.subheader("Select columns for causal estimate")

all_cols = list(df.columns)

treatment_col = st.selectbox("Treatment (must be binary 0/1)", all_cols)
outcome_col = st.selectbox("Outcome (numeric)", all_cols, index=min(1, len(all_cols) - 1))

possible_confounders = [c for c in all_cols if c not in [treatment_col, outcome_col]]
confounders = st.multiselect("Confounders (optional)", possible_confounders, default=[])


def compute_ate_linear(data: pd.DataFrame, treat: str, outcome: str, confs: list[str]):
    """
    ATE = coefficient of treatment in linear regression: outcome ~ treatment + confounders
    """
    cols_x = [treat] + confs
    X = data[cols_x].values
    y = data[outcome].values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    if X.shape[0] == 0:
        return np.nan
    model = LinearRegression()
    model.fit(X, y)
    ate = model.coef_[0]  # treatment is first
    return ate


if st.button("Compute ATE on current data"):
    ate_val = compute_ate_linear(df, treatment_col, outcome_col, confounders)
    st.success(f"Current ATE (treatment coefficient) = **{ate_val:.4f}**")

# ------------------------------------------------------------
# 3) DESIRED ATE RANGE
# ------------------------------------------------------------
st.subheader("Causal repair")

current_ate = compute_ate_linear(df, treatment_col, outcome_col, confounders)
st.write(f"Current ATE: **{current_ate:.4f}**")

desired_center = st.number_input(
    "Desired ATE center (target)", value=float(np.round(current_ate, 3))
)
tolerance = st.number_input("Tolerance (±)", value=0.5, min_value=0.0, step=0.1)

# ------------------------------------------------------------
# 4) CHOOSE ALGORITHM (from the article: tuple-level / pattern-level)
# ------------------------------------------------------------
algorithm = st.selectbox(
    "Choose repair algorithm",
    [
        "Tuple-level (remove individual rows)",
        "Pattern-level (remove subpopulation)",
    ],
)

max_removals = st.slider("Max tuples/patterns to try", min_value=1, max_value=100, value=20)


def in_range(val, c, tol):
    return (val >= c - tol) and (val <= c + tol)


# ------------------------------------------------------------
# tuple-level implementation (your original one)
# ------------------------------------------------------------
def run_tuple_repair(df_input: pd.DataFrame):
    df_work = df_input.copy()
    removed_rows = []

    current = compute_ate_linear(df_work, treatment_col, outcome_col, confounders)
    if in_range(current, desired_center, tolerance):
        return df_work, removed_rows, current

    for step in range(max_removals):
        best_idx = None
        best_ate = None
        best_dist = None

        for idx in df_work.index.tolist():
            tmp = df_work.drop(index=idx)
            ate_tmp = compute_ate_linear(tmp, treatment_col, outcome_col, confounders)
            if np.isnan(ate_tmp):
                continue
            # if already in range, take it
            if in_range(ate_tmp, desired_center, tolerance):
                best_idx = idx
                best_ate = ate_tmp
                best_dist = 0.0
                break
            dist = abs(ate_tmp - desired_center)
            if (best_dist is None) or (dist < best_dist):
                best_dist = dist
                best_idx = idx
                best_ate = ate_tmp

        if best_idx is None:
            break

        removed_rows.append(df_work.loc[best_idx])
        df_work = df_work.drop(index=best_idx)
        current = best_ate

        if in_range(current, desired_center, tolerance):
            break

    return df_work, removed_rows, current


# ------------------------------------------------------------
# pattern-level implementation (simple version)
# ------------------------------------------------------------
def run_pattern_repair(df_input: pd.DataFrame):
    """
    Very simple pattern deletion:
    - consider all columns except treatment/outcome
    - for each distinct value in that column, simulate removing all rows with that value
    - choose the one that brings ATE closest to target
    - repeat up to max_removals or until we're in range

    This mimics "pattern-level" / subpopulation removal from the paper.
    """
    df_work = df_input.copy()
    removed_patterns = []
    current = compute_ate_linear(df_work, treatment_col, outcome_col, confounders)
    other_cols = [c for c in df_work.columns if c not in [treatment_col, outcome_col]]

    if in_range(current, desired_center, tolerance):
        return df_work, removed_patterns, current

    for _ in range(max_removals):
        best_col = None
        best_val = None
        best_ate = None
        best_dist = None

        # search 1-attribute patterns
        for col in other_cols:
            # to keep it small, cap number of unique values per column
            unique_vals = df_work[col].unique()
            if len(unique_vals) > 30:  # just to avoid crazy loops
                unique_vals = unique_vals[:30]

            for v in unique_vals:
                mask = df_work[col] == v
                tmp = df_work[~mask]
                if tmp.empty:
                    continue
                ate_tmp = compute_ate_linear(tmp, treatment_col, outcome_col, confounders)
                if np.isnan(ate_tmp):
                    continue
                if in_range(ate_tmp, desired_center, tolerance):
                    best_col = col
                    best_val = v
                    best_ate = ate_tmp
                    best_dist = 0.0
                    break
                dist = abs(ate_tmp - desired_center)
                if (best_dist is None) or (dist < best_dist):
                    best_dist = dist
                    best_col = col
                    best_val = v
                    best_ate = ate_tmp
            if best_dist == 0.0:
                break

        if best_col is None:
            break

        # apply the best pattern deletion
        removed_patterns.append((best_col, best_val, (df_work[best_col] == best_val).sum()))
        df_work = df_work[df_work[best_col] != best_val]
        current = best_ate

        if in_range(current, desired_center, tolerance):
            break

    return df_work, removed_patterns, current


run_repair = st.button("Run repair")

if run_repair:
    if algorithm.startswith("Tuple-level"):
        df_new, removed, new_ate = run_tuple_repair(df)
        st.success(f"New ATE after **tuple-level** repair: **{new_ate:.4f}**")
        st.write(f"Total tuples removed: **{len(removed)}**")
        if len(removed) > 0:
            st.subheader("Removed tuples")
            st.dataframe(pd.DataFrame(removed))
    else:
        df_new, removed_patterns, new_ate = run_pattern_repair(df)
        st.success(f"New ATE after **pattern-level** repair: **{new_ate:.4f}**")
        st.write(f"Total patterns removed: **{len(removed_patterns)}**")
        if len(removed_patterns) > 0:
            st.subheader("Removed subpopulations")
            st.dataframe(
                pd.DataFrame(removed_patterns, columns=["column", "value", "rows_removed"])
            )

    st.subheader("Repaired dataset head")
    st.dataframe(df_new.head())

    st.info(
        "Note: tuple-level = your original O(n²) demo. "
        "Pattern-level here uses only single-attribute patterns to stay simple; "
        "the paper does smarter multi-predicate random walks."
    )
