import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="E-Commerce Customer Behaviour Analysis",
    layout="wide"
)

# Load dataset
df = pd.read_csv("data/cleaned_data.csv")

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

# App title
st.title("E-Commerce Customer Behaviour Analysis Dashboard")

st.markdown("""
This dashboard analyzes customer purchasing behavior and predicts customer purchase intention using Machine Learning.
""")

# Sidebar
st.sidebar.header("Customer Input")

# Input fields
age = st.sidebar.slider(
    "Age",
    18,
    70,
    25
)

purchase_amount = st.sidebar.number_input(
    "Purchase Amount",
    min_value=0.0,
    value=100.0
)

income_level = st.sidebar.number_input(
    "Income Level",
    min_value=0.0,
    value=5000.0
)

engagement_ads = st.sidebar.slider(
    "Engagement With Ads",
    0,
    10,
    5
)

# Create engineered feature
spending_per_age = purchase_amount / age

# Prediction section
st.subheader("Purchase Prediction")

if st.button("Predict Purchase Intention"):

    # Create full feature input
    sample = df.drop("Purchase_Intent", axis=1).iloc[[0]].copy()

    # Replace important fields
    if "Age" in sample.columns:
        sample["Age"] = age

    if "Purchase_Amount" in sample.columns:
        sample["Purchase_Amount"] = purchase_amount

    if "Income_Level" in sample.columns:
        sample["Income_Level"] = income_level

    if "Engagement_with_Ads" in sample.columns:
        sample["Engagement_with_Ads"] = engagement_ads

    if "Spending_Per_Age" in sample.columns:
        sample["Spending_Per_Age"] = spending_per_age

    # Prediction
    prediction = model.predict(sample)

    st.success(
        f"Predicted Purchase Intention: {prediction[0]}"
    )

# -------------------------------
# Interactive EDA
# -------------------------------

st.subheader("Interactive Exploratory Data Analysis")

# Dataset preview
if st.checkbox("Show Dataset"):

    st.dataframe(df.head())

# Select numerical column
numeric_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns

selected_col = st.selectbox(
    "Select Numerical Feature",
    numeric_cols
)

# Histogram
fig, ax = plt.subplots(figsize=(8,5))

sns.histplot(
    df[selected_col],
    kde=True,
    ax=ax
)

ax.set_title(f"{selected_col} Distribution")

st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(10,8))

sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax2
)

st.pyplot(fig2)

# Scatter plot
st.subheader("Feature Relationship")

x_axis = st.selectbox(
    "Select X-axis",
    numeric_cols,
    key="x"
)

y_axis = st.selectbox(
    "Select Y-axis",
    numeric_cols,
    key="y"
)

fig3, ax3 = plt.subplots(figsize=(8,5))

sns.scatterplot(
    x=df[x_axis],
    y=df[y_axis],
    ax=ax3
)

ax3.set_title(f"{x_axis} vs {y_axis}")

st.pyplot(fig3)

# Footer
st.markdown("---")

st.markdown(
    "Developed for E-Commerce Customer Behaviour Analysis Project"
)