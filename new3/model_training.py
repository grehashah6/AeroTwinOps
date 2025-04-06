# dynamic_model_training.py
import pandas as pd
import time
import os
import joblib
from sklearn.linear_model import LinearRegression

def retrain_model():
    data = pd.read_csv("scenario_data_engineered.csv")
    
    # Define the features used during training.
    # (For simplicity, we use the same subset as in your scenario evaluation.)
    features = ["machine_count", "avg_T_in", "std_T_in", "avg_T_out", "std_T_out",
                "avg_RPM", "std_RPM", "avg_Vibration", "std_Vibration", "cycle_time",
                "energy_consumption", "estimated_travel_distance"]
    X = data[features]
    y = data["throughput"]
    
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "throughput_model.pkl")
    print("Model retrained and saved to throughput_model.pkl.")

if __name__ == "__main__":
    # Watch the engineered data file for changes
    last_modified = os.path.getmtime("scenario_data_engineered.csv")
    print("Monitoring 'scenario_data_engineered.csv' for changes...")
    
    while True:
        current_modified = os.path.getmtime("scenario_data_engineered.csv")
        if current_modified != last_modified:
            print("Engineered data updated; retraining model...")
            retrain_model()
            last_modified = current_modified
        time.sleep(60)  # Check every minute (adjust interval as needed)
