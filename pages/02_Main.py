# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import sys
parent_dir =os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0,parent_dir)

from utils import (
    load_scenario1_data,
    load_scenario2_data,
    load_scenario3_data,
    perform_predictive_analysis
)
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Digital Twin Analysis", layout="wide")
st.title("Digital Twin Simulation & Predictive Analysis")

# Sidebar - Scenario and Component selection
st.sidebar.header("Settings")
scenario = st.sidebar.selectbox("Select Scenario:", ["Scenario 1", "Scenario 2", "Scenario 3"])
component = st.sidebar.selectbox("Select System Component:", ["Hydraulic Pump", "Tanks", "Engines", "Pumps"])

# Load the appropriate data
@st.cache_data
def load_data(scenario):
    if scenario == "Scenario 1":
        return load_scenario1_data()
    elif scenario == "Scenario 2":
        return load_scenario2_data()
    elif scenario == "Scenario 3":
        return load_scenario3_data()

data = load_data(scenario)
if data is None or data.empty:
    st.error("No data available. Please check your CSV files.")
    st.stop()

# Filter data based on component
comp_lower = component.lower()
if comp_lower == "hydraulic pump":
    filtered_data = data[data['source'].str.lower().str.contains("phydraulique")]
elif comp_lower == "tanks":
    filtered_data = data[data['source'].str.lower().str.contains("tank")]
elif comp_lower == "engines":
    filtered_data = data[data['source'].str.lower().str.contains("driver")]
elif comp_lower == "pumps":
    filtered_data = data[data['source'].str.lower().str.contains("pump") & ~data['source'].str.lower().str.contains("phydraulique")]
else:
    filtered_data = data

if filtered_data.empty:
    st.warning("No data found for the selected system component.")
    st.stop()

# Buttons
if st.button("Run Predictive Analysis"):
    report, accuracy, feature_names, importances = perform_predictive_analysis(filtered_data)
    st.subheader("Classification Report")
    st.text(report)
    st.markdown(f"**Accuracy:** {accuracy:.4f}")

    # Plot feature importances
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices[:20]]
    sorted_importances = importances[indices[:20]]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_importances, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top Feature Importances for {component}")
    st.pyplot(fig)

# Additional Features
if st.button("Show Time Series"):
    raw_file = filtered_data['source'].iloc[0]
    try:
        file_path = os.path.join("analysis", raw_file)
        raw_df = pd.read_csv(file_path)
        col = raw_df.select_dtypes(include=[np.number]).columns[0]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(raw_df[col])
        ax.set_title(f"Time Series Plot: {col} – {raw_file}")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading file: {e}")


if st.button("Show Correlation Matrix"):
    corr = filtered_data.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    st.pyplot(fig)

if st.button("Run Anomaly Detection"):
    X = filtered_data.drop(columns=['label', 'source'])
    model = IsolationForest(contamination=0.1)
    preds = model.fit_predict(X)
    filtered_data['anomaly'] = preds

    normal = filtered_data[filtered_data['anomaly'] == 1]
    anomaly = filtered_data[filtered_data['anomaly'] == -1]
    col = X.columns[0]
    fig, ax = plt.subplots()
    ax.plot(normal.index, normal[col], 'b.', label='Normal')
    ax.plot(anomaly.index, anomaly[col], 'ro', label='Anomaly')
    ax.set_title(f"Anomaly Detection - {col}")
    ax.legend()
    st.pyplot(fig)
