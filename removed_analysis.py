import pandas as pd
import numpy as np
from API_KEY import API_KEY
from openai import OpenAI
# import openai  # only needed if you use openai.api_key

# Set API key from Streamlit secrets or env:
# openai.api_key = API_KEY  # st.secrets["OPENAI_KEY"]

client = OpenAI(api_key=API_KEY)


def compute_removed_analysis(removed_df: pd.DataFrame, full_df: pd.DataFrame):
    """Compute key statistics for removed tuples."""

    if removed_df.empty:
        return {"error": "No tuples removed"}, None

    # Numeric columns only
    num_cols = removed_df.select_dtypes(include=np.number).columns

    summary_removed = removed_df[num_cols].describe().T

    # Differences vs global
    summary_full = full_df[num_cols].describe().T
    summary_diff = summary_removed[["mean"]] - summary_full[["mean"]]
    summary_diff.columns = ["mean_shift_vs_full"]

    # ATE differences (if exist)
    for col in ["treatment", "outcome"]:
        if col not in removed_df.columns:
            continue

    # Example simple ATE calculation
    def compute_ate(df):
        if "treatment" in df.columns and "outcome" in df.columns:
            return df[df.treatment == 1].outcome.mean() - df[df.treatment == 0].outcome.mean()
        return None

    ate_removed = compute_ate(removed_df)
    ate_full = compute_ate(full_df)
    ate_diff = None
    if ate_removed is not None and ate_full is not None:
        ate_diff = ate_removed - ate_full

    return {
               "summary_removed": summary_removed,
               "summary_diff": summary_diff,
               "ate_removed": ate_removed,
               "ate_full": ate_full,
               "ate_diff": ate_diff
           }, summary_diff


def call_gpt_for_removed_analysis(summary, diff):
    """Call GPT to generate insights about removed tuples."""

    if "error" in summary:
        return "No removed tuples to analyze."

    prompt = f"""
You are assisting in a causal inference auditing task (SubCure Tuple Removal).
We have identified a subpopulation of removed tuples.

Here is statistical information:

Removed subpopulation summary:
{summary["summary_removed"].to_csv()}

Mean shifts vs full dataset:
{summary["summary_diff"].to_csv()}

ATE full dataset: {summary["ate_full"]}
ATE removed subset: {summary["ate_removed"]}
ATE difference (removed - full): {summary["ate_diff"]}

Please produce a concise analysis describing:

1. What characterizes the removed subpopulation?
2. Which features shift the most?
3. How does the treatment–outcome relationship differ in the removed subset?
4. Potential fairness or bias implications.
5. If the removed subgroup is meaningful (e.g., vulnerable, extreme, influential patterns).

Give a structured bullet-point summary.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content
