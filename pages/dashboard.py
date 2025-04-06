import streamlit as st
import pandas as pd
import numpy as np

# ==============================
# SIMULATED DATA SETUP
# ==============================
np.random.seed(42)
# Create 200 hourly records for more data variation
dates = pd.date_range(start='2025-04-01', periods=200, freq='H')
data = {
    'timestamp': dates,
    'machine': np.random.choice(['Machine A', 'Machine B', 'Machine C'], size=len(dates)),
    'event': np.random.choice(['production', 'breakdown', 'maintenance'], size=len(dates)),
    'temperature': np.random.uniform(60, 100, size=len(dates)),
    'vibration': np.random.uniform(0.1, 1.0, size=len(dates))
}
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ==============================
# DASHBOARD TITLE
# ==============================
st.title("Enhanced Simulation Dashboard")

# ==============================
# SIDEBAR FILTERS
# ==============================
st.sidebar.header("Filter Options")

# Filter 1: Date Range Filter
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
if start_date and end_date:
    df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

# Filter 2: Event Type Filter
available_events = df['event'].unique()
selected_events = st.sidebar.multiselect("Select Event Type(s)", options=available_events, default=list(available_events))
df = df[df['event'].isin(selected_events)]

# Filter 3: Temperature Range Filter
min_temp = float(df['temperature'].min())
max_temp = float(df['temperature'].max())
temp_range = st.sidebar.slider("Select Temperature Range", min_temp, max_temp, (min_temp, max_temp))
df = df[(df['temperature'] >= temp_range[0]) & (df['temperature'] <= temp_range[1])]

# Additional Filter 4: Vibration Range Filter
min_vib = float(df['vibration'].min())
max_vib = float(df['vibration'].max())
vib_range = st.sidebar.slider("Select Vibration Range", min_vib, max_vib, (min_vib, max_vib))
df = df[(df['vibration'] >= vib_range[0]) & (df['vibration'] <= vib_range[1])]

# Additional Filter 5: Machine Selection Filter
available_machines = df['machine'].unique()
selected_machines = st.sidebar.multiselect("Select Machine(s)", options=available_machines, default=list(available_machines))
df = df[df['machine'].isin(selected_machines)]

# ==============================
# MAIN DASHBOARD CONTENT
# ==============================
st.subheader("Filtered Data")
st.write(df)

# ----- KEY METRICS -----
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Events", len(df))
col2.metric("Avg Temperature", f"{df['temperature'].mean():.1f}")
col3.metric("Avg Vibration", f"{df['vibration'].mean():.2f}")

# ----- CHARTS -----

# Chart 1: Temperature over Time
st.subheader("Temperature over Time")
temp_chart = df.set_index('timestamp')['temperature']
st.line_chart(temp_chart)

# Chart 2: Vibration over Time
st.subheader("Vibration over Time")
vib_chart = df.set_index('timestamp')['vibration']
st.line_chart(vib_chart)

# Chart 3: Event Count per Machine
st.subheader("Event Count per Machine")
event_counts = df.groupby('machine')['event'].count().reset_index(name='count')
st.bar_chart(event_counts.set_index('machine'))

# Chart 4: Event Type Distribution
st.subheader("Event Type Distribution")
event_distribution = df['event'].value_counts().reset_index()
event_distribution.columns = ['event', 'count']
st.bar_chart(event_distribution.set_index('event'))

# Optional: You can also add more visualizations, such as scatter plots or pie charts,
# depending on what insights you want to extract from the simulation data.
