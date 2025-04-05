import pandas as pd
import os

# Determine the log file path relative to this file
LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulation_log.csv')

# Initialize the log file if it doesn't exist
try:
    log_df = pd.read_csv(LOG_FILE)
except FileNotFoundError:
    log_df = pd.DataFrame(columns=['timestamp', 'machine', 'event', 'sensor_data'])
    log_df.to_csv(LOG_FILE, index=False)

def log_event(timestamp, machine, event, sensor_data=None):
    global log_df
    new_entry = {
        'timestamp': timestamp,
        'machine': machine,
        'event': event,
        'sensor_data': sensor_data
    }
    # Append the new entry using pd.concat instead of .append()
    log_df = pd.concat([log_df, pd.DataFrame([new_entry])], ignore_index=True)
    log_df.to_csv(LOG_FILE, index=False)
