# dynamic_scenario_evaluation.py
import pandas as pd
import joblib

def evaluate_scenario(scenario):
    # Load the most recent model.
    model = joblib.load("throughput_model.pkl")
    scenario_df = pd.DataFrame(scenario)
    predicted = model.predict(scenario_df)
    return predicted[0]

if __name__ == "__main__":
    # Define the features in the same order as used during training.
    features = ["machine_count", "avg_T_in", "std_T_in", "avg_T_out", "std_T_out",
                "avg_RPM", "std_RPM", "avg_Vibration", "std_Vibration", "cycle_time",
                "energy_consumption", "estimated_travel_distance"]
    
    scenario = {}
    print("Enter scenario values:")
    for feature in features:
        value = float(input(f"  {feature}: "))
        scenario[feature] = [value]
    
    throughput = evaluate_scenario(scenario)
    print(f"Predicted Throughput for the scenario: {throughput}")
