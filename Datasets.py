import streamlit as st
import pandas as pd
import numpy as np


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
