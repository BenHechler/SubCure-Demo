import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def compute_ate_linear(data, treat, outcome, confs):
    X = data[[treat] + confs].values
    y = data[outcome].values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]
    if len(y) == 0:
        return np.nan
    model = LinearRegression().fit(X, y)
    return model.coef_[0]


def in_range(v, c, tol):
    return (c - tol) <= v <= (c + tol)


def run_tuple_repair(df_in, treatment_col, outcome_col, confounders, max_removals, desired_center, tolerance):
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


def run_pattern_repair(df_in, treatment_col, outcome_col, confounders, max_removals, desired_center, tolerance):
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
