


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Australian Weather Data Explorer")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("weatherAUS.csv")

df = load_data()

st.subheader("Raw Data Preview")
st.dataframe(df.head(10))

st.subheader("Dataset Info")
buffer = df.info(buf=None)
st.text("See console/logs for info()")  # Streamlit doesn't display info() directly

st.subheader("Summary Statistics")
st.write(df.describe())

# Optional: Visualization
st.subheader("Rainfall Histogram")
fig, ax = plt.subplots()
sns.histplot(df['Rainfall'].dropna(), kde=True, bins=30, ax=ax)
st.pyplot(fig)

