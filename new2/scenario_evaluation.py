# scenario_evaluation.py
import pandas as pd
import joblib

# Step 1: Load the Trained Model
model = joblib.load("throughput_model.pkl")

# Step 2: Define the New Scenario
# Make sure the feature names and order match those used in training.
new_scenario = {
    "machine_count": [3],
    "avg_T_in": [25.0],
    "std_T_in": [2.0],
    "avg_T_out": [400.0],
    "std_T_out": [15.0],
    "avg_RPM": [20000],
    "std_RPM": [500],
    "avg_Vibration": [0.3],
    "std_Vibration": [0.05],
    "cycle_time": [50], 
    "energy_consumption": [20],
    "estimated_travel_distance": [120]
}

# Convert the scenario into a DataFrame
new_scenario_df = pd.DataFrame(new_scenario)
print("New Scenario DataFrame:")
print(new_scenario_df)

# Step 3: Predict the Performance (Throughput)
predicted_throughput = model.predict(new_scenario_df)
print("Predicted Throughput for the new scenario:", predicted_throughput[0])
