import streamlit as st
import pandas as pd
import os

# Use st.cache_data (if you're using Streamlit 1.18+). Otherwise, use st.cache.
@st.cache_data
def load_data(filename: str):
    """
    Load a CSV file from the simulation/data directory.
    """
    # Get the absolute path to the folder where dashboard.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the CSV file
    data_file_path = os.path.join(current_dir, "simulation", "data", filename)
    
    # Read the CSV into a DataFrame
    df = pd.read_csv(data_file_path)
    return df

def main():
    st.title("Simulation Log Dashboard")

    # Let user pick which CSV file to view
    st.sidebar.header("Select Log File")
    csv_files = ["simulation_log.csv", "simulation_log1.csv"]  # Add more if needed
    selected_csv = st.sidebar.selectbox("Log File", csv_files)

    # Load the selected data
    data = load_data(selected_csv)

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter Logs")

    # Example filters based on assumed columns: 'machine' and 'event'
    if "machine" in data.columns:
        machines = data["machine"].dropna().unique().tolist()
        selected_machines = st.sidebar.multiselect("Select Machines", machines, default=machines)
    else:
        selected_machines = None

    if "event" in data.columns:
        events = data["event"].dropna().unique().tolist()
        selected_events = st.sidebar.multiselect("Select Events", events, default=events)
    else:
        selected_events = None

    # Apply filters (only if columns exist)
    filtered_data = data.copy()
    if selected_machines is not None:
        filtered_data = filtered_data[filtered_data["machine"].isin(selected_machines)]
    if selected_events is not None:
        filtered_data = filtered_data[filtered_data["event"].isin(selected_events)]

    # --- TABS FOR TABLE VIEW & CHART VIEW ---
    tab1, tab2 = st.tabs(["Table View", "Chart View"])

    with tab1:
        st.subheader("Filtered Log Data")
        st.dataframe(filtered_data)

    with tab2:
        st.subheader("Event Counts by Machine")
        if "machine" in filtered_data.columns and "event" in filtered_data.columns:
            event_counts = filtered_data.groupby(["machine", "event"]).size().reset_index(name="count")
            pivot_table = event_counts.pivot(index="machine", columns="event", values="count").fillna(0)
            st.bar_chart(pivot_table)
        else:
            st.write("No 'machine' or 'event' column found in the data to chart.")

    st.markdown("### Additional Features")
    st.write("You can add more widgets, like date-range filters or keyword searches, to further refine your logs.")

if __name__ == "__main__":
    main()
