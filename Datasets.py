import streamlit as st
import pandas as pd
from pathlib import Path


@st.cache_data
def load_twins_dataset():
    treatment = "treatment"
    outcome = "outcome"
    df = load_dataset("twins", treatment, outcome)
    df = df.rename(columns={treatment: "Heavier Twin", outcome: "Mortality"})
    return df


@st.cache_data
def load_ACS_dataset():
    treatment = "With a disability"
    outcome = "Wages or salary income past 12 months"
    df = load_dataset("acs", treatment, outcome)
    del df['Unnamed: 0']
    df = df.rename(columns={
        "Wages or salary income past 12 months": "Wages",
        "With a disability": "No disability",
        "Educational attainment": "Education",
        "Public health coverage": "Public insurance",
        "Private health insurance coverage": "Private insurance",
        "Medicare, for people 65 and older, or people with certain disabilities": "Medicare",
        "Insurance through a current or former employer or union": "Work insurance"
    })
    df["No disability"] = df["No disability"].map({1: 0, 2: 1})
    return df


@st.cache_data
def load_SO_dataset():
    treatment = "FormalEducation"
    outcome = "ConvertedSalary"
    return load_dataset("stackoverflow", treatment, outcome)


@st.cache_data
def load_german_dataset():
    treatment = "housing"
    outcome = "credit_risk"
    df = load_dataset("german_credit", treatment, outcome)
    return df


def load_dataset(name, treatment, outcome):
    root = Path(__file__).resolve().parent
    data_dir = root / "Data"
    name += ".csv"
    full_path = data_dir / name
    df = pd.read_csv(full_path).head(1000)
    cols = [treatment, outcome] + [c for c in df.columns if c not in [treatment, outcome]]
    df = df[cols]
    return df



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

