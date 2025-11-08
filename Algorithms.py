import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import random
from collections import defaultdict


def estimate_ate_linear(df, treatment, outcome, confounders):
    X = df[[treatment] + confounders].to_numpy()
    y = df[outcome].to_numpy()
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]
    if len(y) == 0:
        return np.nan
    model = LinearRegression().fit(X, y)
    return model.coef_[0]  # treatment is first


def in_target(ate, target, eps):
    return (target - eps) <= ate <= (target + eps)


def subcure_tuple(
    df,
    treatment,
    outcome,
    confounders,
    ate_target,
    eps,
    max_iters=200,
    reps_per_cluster=2,
    random_state=0,
):
    # ----- 1) initial ATE
    cur_ate = estimate_ate_linear(df, treatment, outcome, confounders)
    if in_target(cur_ate, ate_target, eps):
        return df, pd.DataFrame([]), cur_ate

    # ----- 2) build clustering space: T, O, Z...
    cols_for_clustering = [treatment, outcome] + confounders
    X = df[cols_for_clustering].to_numpy()

    n = len(df)
    k = max(5, min(int(np.sqrt(n)), n // 10))  # as in paper 5 ≤ k ≤ n/10 approx. :contentReference[oaicite:6]{index=6}
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    cluster_labels = km.fit_predict(X)

    # group indices by cluster
    clusters = {c: np.where(cluster_labels == c)[0].tolist() for c in range(k)}

    # ----- 3) pick representatives per cluster (closest to centroid + 1 more) :contentReference[oaicite:7]{index=7}
    reps_by_cluster = {}
    for c in range(k):
        idxs = clusters[c]
        if len(idxs) <= reps_per_cluster:
            reps_by_cluster[c] = idxs
        else:
            # compute distances to centroid
            centroid = km.cluster_centers_[c]
            dists = [(i, np.linalg.norm(X[i] - centroid)) for i in idxs]
            dists.sort(key=lambda x: x[1])
            # first rep: closest
            chosen = [dists[0][0]]
            # second rep: pick about 75th percentile distance
            if reps_per_cluster > 1:
                second = dists[min(len(dists)-1, int(0.75*len(dists)) )][0]
                chosen.append(second)
            reps_by_cluster[c] = chosen

    # we'll track removed rows
    removed_rows = []

    # ----- 4) iterative, cluster-aware removal
    active_idx = set(range(n))

    for it in range(max_iters):
        # recompute current ATE
        cur_df = df.iloc[list(active_idx)]
        cur_ate = estimate_ate_linear(cur_df, treatment, outcome, confounders)
        if in_target(cur_ate, ate_target, eps):
            break

        # For each cluster, sample up to 5 points that are still active, score them. :contentReference[oaicite:8]{index=8}
        best_candidate = None
        best_new_ate = None
        best_delta = None

        for c in range(k):
            # active points in this cluster
            cluster_active = [i for i in clusters[c] if i in active_idx]
            if not cluster_active:
                continue
            m_k = min(5, len(cluster_active))  # as per paper
            sampled = random.sample(cluster_active, m_k)

            for idx in sampled:
                # remove idx and get new ATE
                tmp_active = active_idx - {idx}
                tmp_df = df.iloc[list(tmp_active)]
                new_ate = estimate_ate_linear(tmp_df, treatment, outcome, confounders)
                # how good is it? measure distance to target center
                delta = abs(new_ate - ate_target)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_candidate = idx
                    best_new_ate = new_ate

        if best_candidate is None:
            # nothing more to do
            break

        # apply best removal
        removed_rows.append(df.iloc[best_candidate])
        active_idx.remove(best_candidate)

    final_df = df.iloc[list(active_idx)]
    final_ate = estimate_ate_linear(final_df, treatment, outcome, confounders)
    removed_df = pd.DataFrame(removed_rows)
    return final_df, removed_df, final_ate


def pattern_tuples(df, pattern):
    """Return mask of rows matching a dict pattern {col: value}."""
    mask = pd.Series(True, index=df.index)
    for col, val in pattern.items():
        mask = mask & (df[col] == val)
    return mask


def subcure_pattern(
    df,
    treatment,
    outcome,
    confounders,
    ate_target,
    eps,
    max_walks=20,
    max_pattern_size=4,
    sample_frac=1.0,
):
    """
    Simplified version of Algorithm 1 in the paper. :contentReference[oaicite:11]{index=11}
    """
    cur_ate = estimate_ate_linear(df, treatment, outcome, confounders)
    if in_target(cur_ate, ate_target, eps):
        # nothing to remove
        return df, pd.DataFrame([]), cur_ate

    # maybe sample to speed up, as they do for ACS :contentReference[oaicite:12]{index=12}
    if sample_frac < 1.0:
        df_work = df.sample(frac=sample_frac, random_state=0)
    else:
        df_work = df

    # 1) get most specific groups = all (col=val) except outcome
    candidate_cols = [c for c in df_work.columns if c not in [outcome]]
    most_specific = []
    for col in candidate_cols:
        for val in df_work[col].unique():
            most_specific.append({col: val})

    # cache of evaluated patterns → ate
    ate_cache = {}

    def eval_after_removal(pattern):
        key = tuple(sorted(pattern.items()))
        if key in ate_cache:
            return ate_cache[key]
        mask = pattern_tuples(df, pattern)
        new_df = df[~mask]
        new_ate = estimate_ate_linear(new_df, treatment, outcome, confounders)
        ate_cache[key] = new_ate
        return new_ate

    # weighting for predicates (dynamic weighting mechanism) :contentReference[oaicite:13]{index=13}
    predicate_weights = defaultdict(lambda: 1.0)

    # try up to k random walks
    for _ in range(max_walks):
        if not most_specific:
            break
        pattern = random.choice(most_specific)
        ate_after = eval_after_removal(pattern)
        if in_target(ate_after, ate_target, eps):
            # found a pattern
            removed_mask = pattern_tuples(df, pattern)
            removed_df = df[removed_mask]
            remaining_df = df[~removed_mask]
            return remaining_df, removed_df, ate_after

        # random walk: keep removing predicates until empty
        current_pattern = pattern.copy()
        while current_pattern:
            # pick a predicate to drop, weighted by past success
            preds = list(current_pattern.items())
            weights = [predicate_weights[(c, v)] for (c, v) in preds]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            # choose predicate index
            chosen_idx = np.random.choice(len(preds), p=probs)
            # drop it
            drop_col, drop_val = preds[chosen_idx]
            new_pattern = {c: v for (c, v) in current_pattern.items() if not (c == drop_col and v == drop_val)}

            ate_after = eval_after_removal(new_pattern)
            # update weight: if got closer, increase
            cur_ate = estimate_ate_linear(df, treatment, outcome, confounders)
            if abs(ate_after - ate_target) < abs(cur_ate - ate_target):
                predicate_weights[(drop_col, drop_val)] *= 1.2  # reward
            else:
                predicate_weights[(drop_col, drop_val)] *= 0.9  # punish

            if in_target(ate_after, ate_target, eps):
                removed_mask = pattern_tuples(df, new_pattern)
                removed_df = df[removed_mask]
                remaining_df = df[~removed_mask]
                return remaining_df, removed_df, ate_after

            # move on
            current_pattern = new_pattern
            if len(current_pattern) > max_pattern_size:
                break  # early terminate like they suggest :contentReference[oaicite:14]{index=14}

    # no solution found
    return df, pd.DataFrame([]), estimate_ate_linear(df, treatment, outcome, confounders)
