# Algorithms.py
import numpy as np
import pandas as pd

from Algo.ATE_update import ATEUpdateLinear
from Algo.clustered_top_k import ClusteredATEBaselines
from Algo.batch_sampled_top_k import BatchSampledATEBaselines


# -------------------------------
# 1. ATE estimation helper
# -------------------------------

def estimate_ate_linear(df: pd.DataFrame,
                        treatment_col: str,
                        outcome_col: str,
                        confounders: list) -> float:
    """
    Estimate the ATE using the exact linear model implementation
    from ATE_update.ATEUpdateLinear.

    Parameters
    ----------
    df : DataFrame
        Full dataset.
    treatment_col : str
        Name of the treatment column (binary 0/1).
    outcome_col : str
        Name of the outcome column (numeric).
    confounders : list of str
        List of confounder columns.

    Returns
    -------
    float
        Original ATE as computed by the SubCuRe linear model.
    """
    if not confounders:
        # Use an empty feature matrix but keep the index aligned
        X = pd.DataFrame(index=df.index)
    else:
        X = df[confounders]

    T = df[treatment_col]
    Y = df[outcome_col]

    ate_model = ATEUpdateLinear(X, T, Y)
    ate = ate_model.get_original_ate()
    return float(ate)


# -------------------------------
# 2. SubCuRe-Tuple (exact)
# -------------------------------

def _run_subcure_tuple_core(X: pd.DataFrame,
                            T: pd.Series,
                            Y: pd.Series,
                            ate_target: float,
                            eps: float,
                            approx: bool = False,
                            influence_recalc_interval: int = 10):
    """
    Internal helper that chooses between ClusteredATEBaselines and
    BatchSampledATEBaselines exactly as in main_subcure_tuple.py.
    It returns the result dict from the underlying implementation.
    """
    model_type = "linear"

    n = len(X)

    if n > 100_000:
        # Large-scale path: BatchSampledATEBaselines + top_k_plus
        # Mirrors the logic in main_subcure_tuple.py
        sample_ratio = 0.1
        batch_size = 400
        k_neighbors = 100

        # We don't rely on a pre-existing sample file in the demo;
        # we allow the sampler to create/use an in-memory sample.
        sampled_baselines = BatchSampledATEBaselines(
            X,
            T,
            Y,
            model_type=model_type,
            sample_ratio=sample_ratio,
            sample_file_path="subcure_sample.csv",
            use_cached_sample=False,
            batch_size=batch_size,
        )

        result = sampled_baselines.batch_sampled_top_k_plus(
            target_ate=ate_target,
            epsilon=eps,
            k_neighbors=k_neighbors,
            approx=approx,
            influence_recalc_interval=influence_recalc_interval,
            verbose=False,
        )

    else:
        # Small/medium path: clustered SubCuRe-Tuple
        sampled_baselines = ClusteredATEBaselines(
            X,
            T,
            Y,
            model_type=model_type,
        )

        result = sampled_baselines.top_k_plus(
            target_ate=ate_target,
            epsilon=eps,
            approx=approx,
            # using clustering-based variant with default clustering args
            use_clustering=True,
            n_clusters=None,
            samples_per_cluster=2,
            influence_recalc_interval=influence_recalc_interval,
            verbose=False,
        )

    return result


def subcure_tuple(df: pd.DataFrame,
                  treatment_col: str,
                  outcome_col: str,
                  confounders: list,
                  ate_target: float,
                  eps: float):
    """
    Tuple-level SubCuRe implementation for the demo.

    This is a thin wrapper around the *exact* SubCuRe-Tuple algorithm
    from the original repository (ClusteredATEBaselines / BatchSampledATEBaselines).

    Parameters
    ----------
    df : DataFrame
        Dataset.
    treatment_col : str
        Treatment column name (binary 0/1).
    outcome_col : str
        Outcome column name (numeric).
    confounders : list of str
        Confounder column names.
    ate_target : float
        Desired ATE center.
    eps : float
        Tolerance around the desired ATE.

    Returns
    -------
    df_new : DataFrame
        Dataset after removing tuples suggested by SubCuRe-Tuple.
    removed_df : DataFrame
        The removed tuples.
    new_ate : float
        The ATE after removal according to the SubCuRe algorithm.
    """
    if not confounders:
        X = pd.DataFrame(index=df.index)
    else:
        X = df[confounders]

    T = df[treatment_col]
    Y = df[outcome_col]

    result = _run_subcure_tuple_core(
        X, T, Y,
        ate_target=ate_target,
        eps=eps,
        approx=False,                # keep exact version
        influence_recalc_interval=10
    )

    removed_indices = result.get("removed_indices", [])
    new_ate = float(result.get("final_ate", np.nan))

    # Build df_new and removed_df from indices
    mask = np.ones(len(df), dtype=bool)
    mask[removed_indices] = False

    df_new = df.loc[mask].reset_index(drop=True)
    removed_df = df.loc[removed_indices].reset_index(drop=True)

    return df_new, removed_df, new_ate


# -------------------------------
# 3. SubCuRe-Pattern (demo-friendly)
# -------------------------------

def subcure_pattern(df: pd.DataFrame,
                    treatment_col: str,
                    outcome_col: str,
                    confounders: list,
                    ate_target: float,
                    eps: float,
                    max_steps: int = 50,
                    min_group_size: int = 10):
    """
    Pattern-level causal repair (demo-friendly version).

    NOTE:
    -----
    The original `subcure_pattern.py` in the repo is written as a
    random-walk script that *prints* statistics and does not return
    the removed tuples. For the interactive Streamlit demo we need
    a clean function that returns:
        (df_new, removed_df, new_ate)

    This function follows the same *spirit*:
        - Remove attribute = value *subpopulations* (patterns),
        - Recompute ATE after each removal,
        - Stop when ATE is within [ate_target - eps, ate_target + eps]
          or when no improvement is possible.

    It greedily chooses, at each step, the pattern that moves the ATE
    closest toward the target.
    """
    work_df = df.copy()
    removed_indices = []

    # For pattern-search we only use confounders as candidate attributes
    pattern_columns = list(confounders)

    if not pattern_columns:
        # If no confounders, fall back to tuple algorithm
        df_new, removed_df, new_ate = subcure_tuple(
            df, treatment_col, outcome_col, confounders, ate_target, eps
        )
        return df_new, removed_df, new_ate

    def _compute_ate(local_df: pd.DataFrame) -> float:
        if not confounders:
            X_local = pd.DataFrame(index=local_df.index)
        else:
            X_local = local_df[confounders]
        T_local = local_df[treatment_col]
        Y_local = local_df[outcome_col]
        model = ATEUpdateLinear(X_local, T_local, Y_local)
        return float(model.get_original_ate())

    current_ate = _compute_ate(work_df)

    # Early exit if already in range
    if ate_target - eps <= current_ate <= ate_target + eps:
        return work_df.reset_index(drop=True), df.iloc[[]].copy(), current_ate

    for step in range(max_steps):
        best_pattern = None
        best_new_ate = None
        best_indices = None
        current_diff = abs(current_ate - ate_target)

        # Search over simple patterns: single attribute=value
        for col in pattern_columns:
            # Skip columns with too many unique values (for speed)
            if work_df[col].nunique() > 50:
                continue

            for val, sub in work_df.groupby(col):
                if len(sub) < min_group_size:
                    continue

                candidate_indices = sub.index
                remaining_df = work_df.drop(index=candidate_indices)

                # Avoid removing everything
                if remaining_df.empty:
                    continue

                candidate_ate = _compute_ate(remaining_df)
                candidate_diff = abs(candidate_ate - ate_target)

                # Must actually move toward the target
                if candidate_diff < current_diff:
                    if (best_new_ate is None) or (candidate_diff < abs(best_new_ate - ate_target)):
                        best_pattern = (col, val)
                        best_new_ate = candidate_ate
                        best_indices = list(candidate_indices)

        # If no improving pattern found, stop
        if best_pattern is None:
            break

        # Apply the best pattern removal
        work_df = work_df.drop(index=best_indices)
        removed_indices.extend(best_indices)
        current_ate = best_new_ate

        # Check stopping rule
        if ate_target - eps <= current_ate <= ate_target + eps:
            break

    # Build outputs
    removed_df = df.loc[removed_indices].reset_index(drop=True)
    df_new = work_df.reset_index(drop=True)

    return df_new, removed_df, current_ate
