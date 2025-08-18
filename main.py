# app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="homepage", layout="wide")

st.title("Our Data")

# --- Data source ---
st.sidebar.header("Data source")
uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    # Replace this with your own DataFrame if you have one:
    # from my_module import my_dataframe as df
    # df = my_dataframe
    # Demo fallback:
    rng = pd.date_range("2024-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "date": rng,
        "series_a": np.random.randn(len(rng)).cumsum(),
        "series_b": np.random.randn(len(rng)).cumsum(),
    })

# --- Column layout (graph left, table right) ---
left, right = st.columns([2, 1], gap="large")

# --- Controls ---
numeric_cols = df.select_dtypes(include="number").columns.tolist()
all_cols = df.columns.tolist()

with st.sidebar:
    st.subheader("Plot settings")
    x_col = st.selectbox("X axis", options=all_cols, index=min(0, len(all_cols)-1))
    y_cols = st.multiselect("Y axis (one or more)", options=numeric_cols, default=numeric_cols[:1])
    show_last_n = st.number_input("Show last N rows", min_value=1, max_value=len(df), value=min(50, len(df)))

df_view = df.tail(int(show_last_n))

# --- Left: chart ---
with left:
    st.markdown("### Chart")
    if x_col in df_view.columns and all(y in df_view.columns for y in y_cols) and len(y_cols) > 0:
        # Ensure x is sorted for nicer plots
        df_plot = df_view.sort_values(by=x_col)
        # Use Streamlit's built-in line chart; you can swap to Altair/Plotly if you prefer
        st.line_chart(df_plot.set_index(x_col)[y_cols])
    else:
        st.info("Select a valid X and at least one numeric Y column in the sidebar.")

# --- Right: table ---
with right:
    st.markdown("### Table")
    st.dataframe(df_view, use_container_width=True)

# Optional: download button
csv = df_view.to_csv(index=False).encode("utf-8")
st.download_button("Download shown table as CSV", data=csv, file_name="table_view.csv", mime="text/csv")
