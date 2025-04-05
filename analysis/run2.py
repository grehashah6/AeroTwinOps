import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Set seed
np.random.seed(42)

# Generate synthetic data
def generate_predictive_maintenance_data(n_samples=1000):
    air_temp = np.random.uniform(20, 100, n_samples)
    process_temp = np.random.uniform(50, 150, n_samples)
    rotational_speed = np.random.uniform(1000, 5000, n_samples)
    torque = np.random.uniform(10, 100, n_samples)
    tool_wear = np.random.uniform(0, 10, n_samples)

    failure_score = (
        (process_temp > 120).astype(int) +
        (rotational_speed > 4000).astype(int) +
        (tool_wear > 7).astype(int)
    )
    machine_failure = (failure_score >= 2).astype(int)

    return pd.DataFrame({
        "AirTemp": air_temp,
        "ProcessTemp": process_temp,
        "RotationalSpeed": rotational_speed,
        "Torque": torque,
        "ToolWear": tool_wear,
        "MachineFailure": machine_failure
    })

def generate_optimization_data(n_samples=1000):
    air_temp = np.random.uniform(20, 100, n_samples)
    process_temp = np.random.uniform(50, 150, n_samples)
    rotational_speed = np.random.uniform(1000, 5000, n_samples)
    torque = np.random.uniform(10, 100, n_samples)
    tool_wear = np.random.uniform(0, 10, n_samples)

    noise = np.random.normal(0, 100, n_samples)
    optimal_speed = 5000 - 0.5 * rotational_speed - 100 * (tool_wear / 10) + noise

    return pd.DataFrame({
        "AirTemp": air_temp,
        "ProcessTemp": process_temp,
        "RotationalSpeed": rotational_speed,
        "Torque": torque,
        "ToolWear": tool_wear,
        "OptimalSpeed": optimal_speed
    })

# Train models and predict
pm_data = generate_predictive_maintenance_data()
opt_data = generate_optimization_data()

# Classifier
X_pm = pm_data.drop(columns=["MachineFailure"])
y_pm = pm_data["MachineFailure"]
X_train_pm, X_test_pm, y_train_pm, y_test_pm = train_test_split(X_pm, y_pm, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_pm, y_train_pm)
y_pred_pm = clf.predict(X_test_pm)
y_proba_pm = clf.predict_proba(X_test_pm)[:, 1]

# Regressor
X_opt = opt_data.drop(columns=["OptimalSpeed"])
y_opt = opt_data["OptimalSpeed"]
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_opt, y_opt, test_size=0.3, random_state=42)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_opt, y_train_opt)
y_pred_opt = reg.predict(X_test_opt)

# Visualize classification probabilities
plt.figure(figsize=(10, 5))
sns.histplot(y_proba_pm, bins=20, kde=True, color='skyblue')
plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold = 0.5')
plt.title("Predictive Maintenance: Probability of Failure")
plt.xlabel("Predicted Probability of Failure")
plt.ylabel("Number of Samples")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize regression results
plt.figure(figsize=(10, 5))
plt.scatter(y_test_opt, y_pred_opt, alpha=0.6, color='green')
plt.plot([min(y_test_opt), max(y_test_opt)], [min(y_test_opt), max(y_test_opt)], 'r--')
plt.title("Optimization Advisor: Predicted vs Actual Optimal Speed")
plt.xlabel("Actual Optimal Speed")
plt.ylabel("Predicted Optimal Speed")
plt.grid(True)
plt.tight_layout()
plt.show()