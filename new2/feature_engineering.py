# feature_engineering.py
import pandas as pd
import matplotlib.pyplot as plt

# Load the aggregated sensor data from the CSV file created in Step 1.
data = pd.read_csv("scenario_data.csv")
print("Initial Aggregated Data:")
print(data.head())

# ============================================================
# 1. Aggregate Sensor Values (already computed in scenario_data.csv)
# ============================================================
# Columns available in data:
# machine_id, machine_count, avg_T_in, std_T_in, avg_T_out, std_T_out,
# avg_RPM, std_RPM, avg_Vibration, std_Vibration, cycle_time,
# throughput, energy_consumption, estimated_travel_distance, timestamp

# ============================================================
# 2. Derived Metrics and Scenario-specific Features
# ============================================================

# Example 1: Efficiency Score (throughput per unit energy consumption)
# Multiplying by 100 to scale the metric
data['efficiency_score'] = data['throughput'] / data['energy_consumption'] * 100

# Example 2: Average Temperature (mean of avg_T_in and avg_T_out)
data['avg_temperature'] = (data['avg_T_in'] + data['avg_T_out']) / 2

# Example 3: Temperature Difference (to see the gap between inlet and outlet)
data['temp_diff'] = data['avg_T_out'] - data['avg_T_in']

# You can also compute additional features such as variability measures,
# e.g., coefficient of variation for temperatures:
data['cv_T_in'] = data['std_T_in'] / data['avg_T_in']
data['cv_T_out'] = data['std_T_out'] / data['avg_T_out']

# ============================================================
# 3. Visualize the Engineered Features (Optional)
# ============================================================

# Plot: Cycle Time vs Throughput
plt.figure(figsize=(8, 5))
plt.scatter(data['cycle_time'], data['throughput'], color='blue', alpha=0.7)
plt.xlabel("Cycle Time (sec)")
plt.ylabel("Throughput (parts/hour)")
plt.title("Cycle Time vs Throughput")
plt.grid(True)
plt.show()

# Plot: Efficiency Score vs Estimated Travel Distance
plt.figure(figsize=(8, 5))
plt.scatter(data['estimated_travel_distance'], data['efficiency_score'], color='green', alpha=0.7)
plt.xlabel("Estimated Travel Distance")
plt.ylabel("Efficiency Score")
plt.title("Efficiency Score vs Estimated Travel Distance")
plt.grid(True)
plt.show()

# ============================================================
# 4. Save the Engineered Data to a New CSV File
# ============================================================
data.to_csv("scenario_data_engineered.csv", index=False)
print("Engineered data saved to scenario_data_engineered.csv")
