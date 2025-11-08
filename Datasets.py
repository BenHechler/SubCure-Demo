import streamlit as st
import pandas as pd
import numpy as np
from dowhy import datasets
# from econml.datasets import dowhy_dataset


@st.cache_data
def load_twins_dataset(seed=42):
    np.random.seed(seed)
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


    # twins = datasets.load_twins()
    # return twins
    # # url = "https://msrmtl-causal-data.s3.amazonaws.com/twins/twins.csv"
    # # return pd.read_csv(url)
    #

@st.cache_data
def load_ACS_dataset():
    return load_twins_dataset(1)
    # url = "https://raw.githubusercontent.com/microsoft/EconML/main/econml/datasets/_data/acs/acs_income.csv"
    # df = pd.read_csv(url)
    # df = df.rename(columns={"college": "treatment", "income": "outcome"})
    # return df


@st.cache_data
def load_SO_dataset():
    return load_twins_dataset(2)

    # url = "https://msrmtl-causal-data.s3.amazonaws.com/stackoverflow/stackoverflow_2019.csv"
    # return pd.read_csv(url)


@st.cache_data
def load_german_dataset():
    return load_twins_dataset(3)

    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    # return pd.read_csv(url, delim_whitespace=True, header=None)


DATASET_META = {
    "Credit": {
        "description": "Credit risk, owning a house → credit outcome.",
        "treatment": "Owning a house",
        "outcome": "Credit risk",
        "confounders": "Personal status, age",
        "org_ate": "0.13",
        "tar_ate": "0 (±0.01)",
        "#tuples": "1,000",
        "#atts": "17",
    },
    "Twins": {
        "description": "Heavier twin → mortality.",
        "treatment": "Heavier twin",
        "outcome": "Mortality",
        "confounders": "Gestational age, birth weight, prenatal care, abnormal amniotic fluid, induced labor, gender, maternal marital status, year of birth, previous deliveries",
        "org_ate": "-0.016",
        "tar_ate": "0 (±0.001)",
        "#tuples": "23,968",
        "#atts": "53",
    },
    "Stack Overflow": {
        "description": "High education → annual salary (Stack Overflow).",
        "treatment": "High Education",
        "outcome": "Annual Salary",
        "confounders": "Continent, gender, ethnicity",
        "org_ate": "13,236",
        "tar_ate": "8,236 (±100)",
        "#tuples": "47,702",
        "#atts": "21",
    },
    "ACS": {
        "description": "Not having a disability → annual wage.",
        "treatment": "Not having a disability",
        "outcome": "Annual wage",
        "confounders": "Education, public health coverage, private health coverage, medicare 65+, insurance through employer, gender, age",
        "org_ate": "8,774",
        "tar_ate": "12,000 (±500)",
        "#tuples": "1,188,308",
        "#atts": "17",
    },
}
