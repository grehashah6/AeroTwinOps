# dynamic_feature_engineering.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

def update_plots():
    data = pd.read_csv("scenario_data.csv")
    # Recompute engineered features
    data['efficiency_score'] = data['throughput'] / data['energy_consumption'] * 100
    data['avg_temperature'] = (data['avg_T_in'] + data['avg_T_out']) / 2
    data['temp_diff'] = data['avg_T_out'] - data['avg_T_in']
    data['cv_T_in'] = data['std_T_in'] / data['avg_T_in']
    data['cv_T_out'] = data['std_T_out'] / data['avg_T_out']
    
    # Clear and update the figures
    plt.clf()
    
    # Use subplots to show multiple graphs in one window
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(data['cycle_time'], data['throughput'], color='blue', alpha=0.7)
    ax1.set_xlabel("Cycle Time (sec)")
    ax1.set_ylabel("Throughput (parts/hour)")
    ax1.set_title("Cycle Time vs Throughput")
    ax1.grid(True)
    
    ax2.scatter(data['estimated_travel_distance'], data['efficiency_score'], color='green', alpha=0.7)
    ax2.set_xlabel("Estimated Travel Distance")
    ax2.set_ylabel("Efficiency Score")
    ax2.set_title("Efficiency Score vs Estimated Travel Distance")
    ax2.grid(True)
    
    # Pause briefly so that the GUI event loop can update
    plt.pause(0.1)

if __name__ == "__main__":
    plt.ion()  # Turn on interactive mode
    # Get the last modification time of the CSV file.
    last_modified = os.path.getmtime("scenario_data.csv")
    print("Monitoring 'scenario_data.csv' for changes...")
    
    try:
        while True:
            current_modified = os.path.getmtime("scenario_data.csv")
            if current_modified != last_modified:
                print("CSV file updated; reloading data and updating plots...")
                update_plots()
                last_modified = current_modified
            time.sleep(5)  # Check every 5 seconds (adjust as needed)
    except KeyboardInterrupt:
        plt.ioff()  # Turn interactive mode off when exiting
        plt.show()
