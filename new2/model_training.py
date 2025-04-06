# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load the Engineered Data
# The engineered CSV was created in Step 2 (feature_engineering.py)
data = pd.read_csv("scenario_data_engineered.csv")
print("Data Head:")
print(data.head())

# Step 2: Prepare Your Dataset: Define Features (X) and Target (y)
# In this example, we'll predict throughput.
# You can adjust the list of features based on your needs.
features = [
    "machine_count",
    "avg_T_in", "std_T_in",
    "avg_T_out", "std_T_out",
    "avg_RPM", "std_RPM",
    "avg_Vibration", "std_Vibration",
    "cycle_time",
    "energy_consumption",
    "estimated_travel_distance"
]
X = data[features]
y = data["throughput"]

# Step 3: Split Data into Training and Test Sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Select and Instantiate a Regression Model
# We use RandomForestRegressor here because it handles non-linear relationships well.
model = RandomForestRegressor(random_state=42)

# Step 5: Train the Model Using the Training Data
model.fit(X_train, y_train)

# Step 6: Evaluate the Model on the Test Set
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Step 7: Save the Trained Model for Later Use
joblib.dump(model, "throughput_model.pkl")
print("Trained model saved as throughput_model.pkl")
