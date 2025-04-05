import streamlit as st
import numpy as np
import pandas as pd

# Title and Description
st.title("AeroTwinOps Dashboard")
st.write("Real-time monitoring of the simulated aerospace factory process.")

kpi_data = {
    "Production Throughput": np.random.randint(80, 120),
    "Machine Uptime (%)": np.random.randint(90, 100),
    "Failures Count": np.random.randint(0, 5)
}

# Create three columns for displaying KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Throughput", kpi_data["Production Throughput"])
col2.metric("Uptime (%)", kpi_data["Machine Uptime (%)"])
col3.metric("Failures", kpi_data["Failures Count"])

# Generate dummy time-series data for visualization
time_series = pd.DataFrame({
    "Time": pd.date_range(start=pd.Timestamp.now(), periods=20, freq="T"),
    "Throughput": np.random.randint(80, 120, 20)
})

# Display the line chart
st.line_chart(time_series.set_index("Time"))

# Scenario Controls in Sidebar
st.sidebar.header("Scenario Controls")
production_speed = st.sidebar.slider("Production Speed", 50, 150, 100)
maintenance_interval = st.sidebar.selectbox("Maintenance Interval (mins)", [30, 60, 90])


# # Refresh Button to simulate data updates
# if st.button("Refresh Data"):
#     st.experimental_rerun()

