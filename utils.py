import os
import pandas as pd
import numpy as np

def extract_features(df, window_size=100, step=50):
    features = []
    for start in range(0, len(df) - window_size + 1, step):
        window = df.iloc[start:start + window_size]
        feat = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                feat[col + '_mean'] = window[col].mean()
                feat[col + '_std'] = window[col].std()
                feat[col + '_min'] = window[col].min()
                feat[col + '_max'] = window[col].max()
        features.append(feat)
    return pd.DataFrame(features)

def load_and_process_file(filepath, label, window_size=100, step=50):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    lower_filepath = filepath.lower()
    rename_map = {}
    if "driver" in lower_filepath:
        rename_map = {'0': 'DriverPower'}
    elif "phydraulique" in lower_filepath:
        rename_map = {'0': 'PumpMotorSpeed'}
    elif "pump" in lower_filepath and "phydraulique" not in lower_filepath:
        rename_map = {'0': 'PumpFlow'}
    elif "tank" in lower_filepath:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            rename_map = {'0': 'TankVolume', '1': 'TankTemperature'}
        else:
            rename_map = {'0': 'TankMeasurement'}

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None

    feature_df = extract_features(numeric_df, window_size, step)
    feature_df['label'] = label
    feature_df['source'] = os.path.basename(filepath)
    return feature_df

def load_scenario_data(files, window_size=100, step=50):
    data_list = []
    for file, label in files:
        if os.path.exists(file):
            df = load_and_process_file(file, label, window_size, step)
            if df is not None:
                data_list.append(df)
        else:
            print(f"File {file} not found.")
    if data_list:
        return pd.concat(data_list, ignore_index=True)
    else:
        return None

def load_scenario1_data():
    files = [
        ("analysis/Healthy_Scenario_Driver.csv", 0),
        ("analysis/Healthy_Scenario_Tank.csv", 0),
        ("analysis/Healthy_Scenario1_Driver.csv", 0),
        ("analysis/Healthy_Scenario1_Tank.csv", 0),
        ("analysis/Healthy_Scenario1_Phydraulique.csv", 0),
        ("analysis/Fault_Type2_Driver.csv", 1),
        ("analysis/Fault_Type2_Tank.csv", 1),
        ("analysis/Fault_Type2_Phydraulique.csv", 1)
    ]
    return load_scenario_data(files)

def load_scenario2_data():
    files = [
        ("analysis/Healthy_Scenario2_Driver.csv", 0),
        ("analysis/Healthy_Scenario2_Phydraulique.csv", 0),
        ("analysis/Healthy_Scenario2_Pump.csv", 0),
        ("analysis/Healthy_Scenario2_Tank.csv", 0),
        ("analysis/Fault_Type3_Driver.csv", 1),
        ("analysis/Fault_Type3_Phydraulique.csv", 1),
        ("analysis/Fault_Type3_Pump.csv", 1),
        ("analysis/Fault_Type3_Tank.csv", 1)
    ]
    return load_scenario_data(files)

def load_scenario3_data():
    files = [
        ("analysis/Healthy_Scenario3_Driver.csv", 0),
        ("analysis/Healthy_Scenario3_Phydraulique.csv", 0),
        ("analysis/Healthy_Scenario3_Pump.csv", 0),
        ("analysis/Healthy_Scenario3_Tank.csv", 0),
        ("analysis/Fault_Type3+4_Driver.csv", 1),
        ("analysis/Fault_Type3+4_Phydraulique.csv", 1),
        ("analysis/Fault_Type3+4_Tank.csv", 1)
    ]
    return load_scenario_data(files)

def perform_predictive_analysis(data):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    X = data.drop(columns=['label', 'source'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    importances = clf.feature_importances_
    feature_names = X.columns
    return report, accuracy, feature_names, importances
