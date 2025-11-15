import streamlit as st
import pandas as pd
from pathlib import Path


@st.cache_data
def load_twins_dataset():
    return load_dataset("twins")


@st.cache_data
def load_ACS_dataset():
    df = load_dataset("acs")
    del df['Unnamed: 0']
    T = "With a disability"
    O = "Wages or salary income past 12 months"
    cols = [T, O] + [c for c in df.columns if c not in [T, O]]
    df = df[cols]
    return df


@st.cache_data
def load_SO_dataset():
    return load_dataset("stackoverflow")


@st.cache_data
def load_german_dataset():
    return load_dataset("german_credit")


def load_dataset(name):
    root = Path(__file__).resolve().parent
    data_dir = root / "Data"
    name += ".csv"
    full_path = data_dir / name
    df = pd.read_csv(full_path).head(1000)
    return df


# DATASET_META = {
#     "Credit": {
#         "description": "Credit risk, owning a house → credit outcome.",
#         "treatment": "Owning a house",
#         "outcome": "Credit risk",
#         "confounders": "Personal status, age",
#         "org_ate": "0.13",
#         "tar_ate": "0 (±0.01)",
#         "#tuples": "1,000",
#         "#atts": "17",
#     },
#     "Twins": {
#         "description": "Heavier twin → mortality.",
#         "treatment": "Heavier twin",
#         "outcome": "Mortality",
#         "confounders": "Gestational age, birth weight, prenatal care, abnormal amniotic fluid, induced labor, gender, maternal marital status, year of birth, previous deliveries",
#         "org_ate": "-0.016",
#         "tar_ate": "0 (±0.001)",
#         "#tuples": "23,968",
#         "#atts": "53",
#     },
#     "Stack Overflow": {
#         "description": "High education → annual salary (Stack Overflow).",
#         "treatment": "High Education",
#         "outcome": "Annual Salary",
#         "confounders": "Continent, gender, ethnicity",
#         "org_ate": "13,236",
#         "tar_ate": "8,236 (±100)",
#         "#tuples": "47,702",
#         "#atts": "21",
#     },
#     "ACS": {
#         "description": "Not having a disability → annual wage.",
#         "treatment": "Not having a disability",
#         "outcome": "Annual wage",
#         "confounders": "Education, public health coverage, private health coverage, medicare 65+, insurance through employer, gender, age",
#         "org_ate": "8,774",
#         "tar_ate": "12,000 (±500)",
#         "#tuples": "1,188,308",
#         "#atts": "17",
#     },
# }
