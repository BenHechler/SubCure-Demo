import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Causal Cardinality Repair Demo", layout="wide")

st.title("Causal ATE Demo (mini SubCure-style)")

st.markdown(
    """
This is a minimal demo inspired by **“Stress-Testing Causal Claims via Cardinality Repairs.”**  
Flow:
1. choose dataset  
2. pick treatment / outcome / confounders  
3. compute ATE  
4. set desired ATE range  
5. run a simple greedy tuple-removal to push ATE into the range
"""
)

# ------------------------------------------------------------
# 1) DATASETS
# ------------------------------------------------------------
@st.cache_data
def load_synthetic_wages():
    # synthetic dataset with binary disability and wages
    np.random.seed(42)
    n = 500
    # treatment: 1 = no disability, 0 = disability
    treatment = np.random.binomial(1, 0.7, size=n)
    edu_years = np.random.randint(8, 20, size=n)
    age = np.random.randint(20, 65, size=n)
    base = 15000 + 1200 * edu_years + 200 * (age - 30)
    treatment_effect = 6000  # people w/o disability earn more
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
def load_synthetic_health():
    # Binary treatment: got_treatment; outcome: recovery_time
    np.random.seed(123)
    n = 400
    got_treatment = np.random.binomial(1, 0.5, size=n)
    severity = np.random.randint(1, 5, size=n)
    age = np.random.randint(18, 80, size=n)
    base = 20 + 3 * severity + 0.1 * (age - 40)
    # treatment decreases recovery time by ~2 days
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
    ["Synthetic Wages", "Synthetic Health", "Upload CSV"],
)

if dataset_name == "Synthetic Wages":
    df = load_synthetic_wages()
elif dataset_name == "Synthetic Health":
    df = load_synthetic_health()
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
    # drop rows with NaNs
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    if X.shape[0] == 0:
        return np.nan
    model = LinearRegression()
    model.fit(X, y)
    # treatment is always the first col
    ate = model.coef_[0]
    return ate


if st.button("Compute ATE on current data"):
    ate_val = compute_ate_linear(df, treatment_col, outcome_col, confounders)
    st.success(f"Current ATE (treatment coefficient) = **{ate_val:.4f}**")

# ------------------------------------------------------------
# 3) DESIRED ATE RANGE + GREEDY REPAIR
# ------------------------------------------------------------
st.subheader("Causal repair (very simple greedy version)")

current_ate = compute_ate_linear(df, treatment_col, outcome_col, confounders)
st.write(f"Current ATE: **{current_ate:.4f}**")

desired_center = st.number_input(
    "Desired ATE center (target)", value=float(np.round(current_ate, 3))
)
tolerance = st.number_input("Tolerance (±)", value=0.5, min_value=0.0, step=0.1)

max_removals = st.slider("Max tuples to try to remove", min_value=1, max_value=100, value=20)

st.markdown(
    "Click to run a tiny greedy search: at each step we remove **one row** that moves ATE closest to the target range."
)

run_repair = st.button("Run repair")

if run_repair:
    df_work = df.copy()
    removed_rows = []

    def in_range(val, c, tol):
        return (val >= c - tol) and (val <= c + tol)

    current_ate = compute_ate_linear(df_work, treatment_col, outcome_col, confounders)

    # stop early if already fine
    if in_range(current_ate, desired_center, tolerance):
        st.info("ATE is already in the desired range — no removal needed.")
    else:
        for step in range(max_removals):
            # for each remaining row, test removing it
            best_idx = None
            best_ate = None
            best_dist = None

            # to keep demo fast, cap per-iteration candidates
            # (for real code, you'd do clustering / influence approximation as in the paper)
            candidate_indices = df_work.index.tolist()

            for idx in candidate_indices:
                tmp = df_work.drop(index=idx)
                ate_tmp = compute_ate_linear(tmp, treatment_col, outcome_col, confounders)
                if np.isnan(ate_tmp):
                    continue
                # distance to target interval (0 if inside)
                if in_range(ate_tmp, desired_center, tolerance):
                    best_idx = idx
                    best_ate = ate_tmp
                    best_dist = 0.0
                    break
                # otherwise distance to center
                dist = abs(ate_tmp - desired_center)
                if (best_dist is None) or (dist < best_dist):
                    best_dist = dist
                    best_idx = idx
                    best_ate = ate_tmp

            # apply the best removal
            if best_idx is not None:
                removed_rows.append(df_work.loc[best_idx])
                df_work = df_work.drop(index=best_idx)
                current_ate = best_ate

                if in_range(current_ate, desired_center, tolerance):
                    break
            else:
                # no improvement
                break

        st.success(f"New ATE after repair: **{current_ate:.4f}**")
        st.write(f"Total tuples removed: **{len(removed_rows)}**")

        if len(removed_rows) > 0:
            removed_df = pd.DataFrame(removed_rows)
            st.subheader("Removed tuples")
            st.dataframe(removed_df)

        st.subheader("Repaired dataset head")
        st.dataframe(df_work.head())

        st.info(
            "Note: this is a *naive* O(n²) greedy demo for small data. "
            "The paper uses smarter clustering + incremental ATE updates to scale."
        )
