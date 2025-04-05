#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#####################################
# Data Processing Functions
#####################################

def extract_features(df, window_size=100, step=50):
    """
    Extracts features from a DataFrame using a sliding window.
    Computes mean, standard deviation, min, and max for each numeric column.
    """
    features = []
    for start in range(0, len(df) - window_size + 1, step):
        window = df.iloc[start:start+window_size]
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
    """
    Loads a CSV file, renames columns based on the file name for better clarity,
    extracts numeric features, and attaches a label.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Determine a renaming map based on the file name
    lower_filepath = filepath.lower()
    rename_map = {}
    if "driver" in lower_filepath:
        # For engines: measured parameter is Driver Power (kW)
        rename_map = {'0': 'DriverPower'}
    elif "phydraulique" in lower_filepath:
        # For hydraulic pumps: measured parameter is Pump Motor Speed (RPM)
        rename_map = {'0': 'PumpMotorSpeed'}
    elif "pump" in lower_filepath and "phydraulique" not in lower_filepath:
        # For pumps: measured parameter is Pump Flow (L/min)
        rename_map = {'0': 'PumpFlow'}
    elif "tank" in lower_filepath:
        # For tanks: measured parameters are Tank Volume and Tank Temperature
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            rename_map = {'0': 'TankVolume', '1': 'TankTemperature'}
        else:
            rename_map = {'0': 'TankMeasurement'}

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print(f"No numeric columns found in {filepath}.")
        return None

    feature_df = extract_features(numeric_df, window_size, step)
    feature_df['label'] = label
    feature_df['source'] = os.path.basename(filepath)
    return feature_df

def load_scenario_data(files, window_size=100, step=50):
    """
    Processes multiple files given as (filename, label) tuples.
    """
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

def load_scenario1_data(window_size=100, step=50):
    files = [
        ("Healthy_Scenario_Driver.csv", 0),
        ("Healthy_Scenario_Tank.csv", 0),
        ("Healthy_Scenario1_Driver.csv", 0),
        ("Healthy_Scenario1_Tank.csv", 0),
        ("Healthy_Scenario1_Phydraulique.csv", 0),
        ("Fault_Type2_Driver.csv", 1),
        ("Fault_Type2_Tank.csv", 1),
        ("Fault_Type2_Phydraulique.csv", 1)
    ]
    data = load_scenario_data(files, window_size, step)
    if data is None:
        print("No Scenario 1 data found. Generating dummy data.")
        dummy = pd.DataFrame({
            'sensor_mean': np.random.rand(50),
            'sensor_std': np.random.rand(50),
            'sensor_min': np.random.rand(50),
            'sensor_max': np.random.rand(50),
            'label': np.zeros(50, dtype=int),
            'source': ['dummy_driver']*50
        })
        data = dummy
    return data

def load_scenario2_data(window_size=100, step=50):
    files = [
        ("Healthy_Scenario2_Driver.csv", 0),
        ("Healthy_Scenario2_Phydraulique.csv", 0),
        ("Healthy_Scenario2_Pump.csv", 0),
        ("Healthy_Scenario2_Tank.csv", 0),
        ("Fault_Type3_Driver.csv", 1),
        ("Fault_Type3_Phydraulique.csv", 1),
        ("Fault_Type3_Pump.csv", 1),
        ("Fault_Type3_Tank.csv", 1)
    ]
    data = load_scenario_data(files, window_size, step)
    if data is None:
        print("No Scenario 2 data found. Generating dummy data.")
        dummy = pd.DataFrame({
            'sensor_mean': np.random.rand(50),
            'sensor_std': np.random.rand(50),
            'sensor_min': np.random.rand(50),
            'sensor_max': np.random.rand(50),
            'label': np.ones(50, dtype=int),
            'source': ['dummy_pump']*50
        })
        data = dummy
    return data

def load_scenario3_data(window_size=100, step=50):
    files = [
        ("Healthy_Scenario3_Driver.csv", 0),
        ("Healthy_Scenario3_Phydraulique.csv", 0),
        ("Healthy_Scenario3_Pump.csv", 0),
        ("Healthy_Scenario3_Tank.csv", 0),
        ("Fault_Type3+4_Driver.csv", 1),
        ("Fault_Type3+4_Phydraulique.csv", 1),
        ("Fault_Type3+4_Tank.csv", 1)
    ]
    data = load_scenario_data(files, window_size, step)
    if data is None:
        print("No Scenario 3 data found. Generating dummy data.")
        dummy = pd.DataFrame({
            'sensor_mean': np.random.rand(50),
            'sensor_std': np.random.rand(50),
            'sensor_min': np.random.rand(50),
            'sensor_max': np.random.rand(50),
            'label': np.concatenate((np.zeros(30, dtype=int), np.ones(20, dtype=int))),
            'source': ['dummy_tank']*50
        })
        data = dummy
    return data

def perform_predictive_analysis(data):
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

#####################################
# UI with Tkinter: Component-Specific Analysis with Additional Features
#####################################

class DigitalTwinUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digital Twin Simulation & Predictive Analysis")
        self.geometry("1000x750")
        self.create_widgets()

    def create_widgets(self):
        # Scenario selection
        scenario_lbl = tk.Label(self, text="Select Scenario:")
        scenario_lbl.pack(pady=5)
        self.scenario_var = tk.StringVar(value="Scenario 1")
        self.scenario_dropdown = ttk.Combobox(self, textvariable=self.scenario_var, state="readonly")
        self.scenario_dropdown['values'] = ["Scenario 1", "Scenario 2", "Scenario 3"]
        self.scenario_dropdown.pack(pady=5)

        # System Component selection
        comp_lbl = tk.Label(self, text="Select System Component:")
        comp_lbl.pack(pady=5)
        self.component_var = tk.StringVar(value="Hydraulic Pump")
        self.component_dropdown = ttk.Combobox(self, textvariable=self.component_var, state="readonly")
        self.component_dropdown['values'] = ["Hydraulic Pump", "Tanks", "Engines", "Pumps"]
        self.component_dropdown.pack(pady=5)

        # Run simulation button (Predictive Analysis)
        run_btn = tk.Button(self, text="Run Simulation", command=self.run_simulation)
        run_btn.pack(pady=10)

        # Text widget for displaying analysis results
        self.output_text = tk.Text(self, height=10, width=100)
        self.output_text.pack(pady=10)

        # Frame for matplotlib plot (Predictive Analysis)
        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Additional Features Frame
        self.additional_frame = tk.Frame(self)
        self.additional_frame.pack(pady=10, fill=tk.BOTH)
        ts_btn = tk.Button(self.additional_frame, text="Time Series Visualization", command=self.show_time_series)
        ts_btn.pack(side=tk.LEFT, padx=5)
        corr_btn = tk.Button(self.additional_frame, text="Correlation Analysis", command=self.show_correlation)
        corr_btn.pack(side=tk.LEFT, padx=5)
        anomaly_btn = tk.Button(self.additional_frame, text="Anomaly Detection", command=self.run_anomaly_detection)
        anomaly_btn.pack(side=tk.LEFT, padx=5)

    def run_simulation(self):
        scenario = self.scenario_var.get()
        component = self.component_var.get()
        self.output_text.delete(1.0, tk.END)
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        if scenario == "Scenario 1":
            data = load_scenario1_data()
        elif scenario == "Scenario 2":
            data = load_scenario2_data()
        elif scenario == "Scenario 3":
            data = load_scenario3_data()
        else:
            data = None

        if data is None or data.empty:
            messagebox.showerror("Error", "No data loaded. Please check your CSV files.")
            return

        # Filter data based on system component
        comp_lower = component.lower()
        if comp_lower == "hydraulic pump":
            filtered_data = data[data['source'].str.lower().str.contains("phydraulique")]
        elif comp_lower == "tanks":
            filtered_data = data[data['source'].str.lower().str.contains("tank")]
        elif comp_lower == "engines":
            filtered_data = data[data['source'].str.lower().str.contains("driver")]
        elif comp_lower == "pumps":
            filtered_data = data[data['source'].str.lower().str.contains("pump") & ~data['source'].str.lower().str.contains("phydraulique")]
        else:
            filtered_data = data

        if filtered_data.empty:
            self.output_text.insert(tk.END, "No data found for the selected system component.\n")
            return

        self.output_text.insert(tk.END, f"Data loaded for {component} in {scenario}.\nStarting predictive analysis...\n")
        report, accuracy, feature_names, importances = perform_predictive_analysis(filtered_data)
        self.output_text.insert(tk.END, f"\nClassification Report:\n{report}\n")
        self.output_text.insert(tk.END, f"Accuracy: {accuracy:.4f}\n")

        # Create a sorted horizontal bar chart for feature importances
        fig, ax = plt.subplots(figsize=(8, 6))
        indices = np.argsort(importances)[::-1]
        sorted_importances = importances[indices]
        sorted_features = [feature_names[i] for i in indices]
        top_n = 20
        if len(sorted_features) > top_n:
            sorted_features = sorted_features[:top_n]
            sorted_importances = sorted_importances[:top_n]
        y_pos = np.arange(len(sorted_features))
        ax.barh(y_pos, sorted_importances, align='center', color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.invert_yaxis()
        ax.set_xlabel("Relative Importance Value\n(This value indicates how much a feature contributes to the prediction)", fontsize=10)
        ax.set_ylabel("Aggregated Feature (Mean, Std, Min, Max)\n(Each feature is a statistical summary of sensor data)", fontsize=10)
        ax.set_title(f"Random Forest Feature Importances - {component}\n(Features sorted by their contribution to fault prediction)", fontsize=11, pad=20)
        
        # Annotation describing the plot
        annotation_text = ""
        if comp_lower == "tanks":
            annotation_text = ("Simulation for Tanks:\nThis plot shows the relative importance of statistical features derived from tank data.\n"
                               "The features (e.g., TankVolume_mean, TankTemperature_std) are computed as the mean, standard deviation,\n"
                               "minimum, and maximum over a time window, and higher values indicate greater importance for fault prediction.")
        elif comp_lower == "hydraulic pump":
            annotation_text = ("Simulation for Hydraulic Pump:\nThis plot shows feature importance based on pump motor speed metrics.\n"
                               "The features (e.g., PumpMotorSpeed_mean) help in diagnosing the operational health of the hydraulic pump.")
        elif comp_lower == "engines":
            annotation_text = ("Simulation for Engines:\nThis plot shows the importance of features derived from engine driver power data.\n"
                               "These features assist in detecting abnormal readings and potential engine faults.")
        elif comp_lower == "pumps":
            annotation_text = ("Simulation for Pumps:\nThis plot shows the relative importance of features derived from pump flow data.\n"
                               "Monitoring these features aids in detecting anomalies and ensuring optimal pump performance.")
        fig.tight_layout(rect=[0, 0.15, 1, 1])
        fig.text(0.5, 0.03, annotation_text, ha='center', fontsize=9, wrap=True)
        ax.tick_params(axis='y', labelsize=8)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_time_series(self):
        # Display a time series plot for one matching raw CSV file.
        scenario = self.scenario_var.get()
        component = self.component_var.get().lower()
        file_candidates = []
        if scenario == "Scenario 1":
            files = [("Healthy_Scenario_Driver.csv",0),("Healthy_Scenario_Tank.csv",0),
                     ("Healthy_Scenario1_Driver.csv",0),("Healthy_Scenario1_Tank.csv",0),
                     ("Healthy_Scenario1_Phydraulique.csv",0),
                     ("Fault_Type2_Driver.csv",1),("Fault_Type2_Tank.csv",1),
                     ("Fault_Type2_Phydraulique.csv",1)]
        elif scenario == "Scenario 2":
            files = [("Healthy_Scenario2_Driver.csv",0),("Healthy_Scenario2_Phydraulique.csv",0),
                     ("Healthy_Scenario2_Pump.csv",0),("Healthy_Scenario2_Tank.csv",0),
                     ("Fault_Type3_Driver.csv",1),("Fault_Type3_Phydraulique.csv",1),
                     ("Fault_Type3_Pump.csv",1),("Fault_Type3_Tank.csv",1)]
        elif scenario == "Scenario 3":
            files = [("Healthy_Scenario3_Driver.csv",0),("Healthy_Scenario3_Phydraulique.csv",0),
                     ("Healthy_Scenario3_Pump.csv",0),("Healthy_Scenario3_Tank.csv",0),
                     ("Fault_Type3+4_Driver.csv",1),("Fault_Type3+4_Phydraulique.csv",1),
                     ("Fault_Type3+4_Tank.csv",1)]
        else:
            files = []
        for f, label in files:
            if os.path.exists(f):
                f_lower = f.lower()
                if component == "hydraulic pump" and "phydraulique" in f_lower:
                    file_candidates.append(f)
                elif component == "tanks" and "tank" in f_lower:
                    file_candidates.append(f)
                elif component == "engines" and "driver" in f_lower:
                    file_candidates.append(f)
                elif component == "pumps" and "pump" in f_lower and "phydraulique" not in f_lower:
                    file_candidates.append(f)
        if not file_candidates:
            messagebox.showerror("Error", "No matching file found for time series visualization.")
            return
        file_to_plot = file_candidates[0]
        try:
            df = pd.read_csv(file_to_plot)
        except Exception as e:
            messagebox.showerror("Error", f"Error reading file: {file_to_plot}\n{e}")
            return
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            messagebox.showerror("Error", "No numeric columns found in file for time series plot.")
            return
        col_to_plot = numeric_cols[0]
        
        # Set y-axis label based on selected component
        if component == "tanks":
            y_label = "Tank Measurement (Volume in Liters / Temperature in °C)"
        elif component == "hydraulic pump":
            y_label = "Hydraulic Pump Measurement (Motor Speed in RPM / Flow Rate in L/min)"
        elif component == "engines":
            y_label = "Engine Measurement (Driver Power in kW)"
        elif component == "pumps":
            y_label = "Pump Measurement (Flow in L/min)"
        else:
            y_label = f"{col_to_plot} Value"
        
        ts_window = tk.Toplevel(self)
        ts_window.title("Time Series Visualization")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df.index, df[col_to_plot], label=col_to_plot, color='purple')
        ax.set_title(f"Time Series of {col_to_plot}\n(from file: {file_to_plot})", fontsize=11)
        ax.set_xlabel("Time (seconds)\n(Each data point represents 1 second)", fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.legend(loc='best')
        ax.text(0.5, 0.02, "This plot shows how the sensor reading changes over time, helping to identify trends and deviations.", 
                transform=ax.transAxes, fontsize=9, ha='center')
        canvas = FigureCanvasTkAgg(fig, master=ts_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_correlation(self):
        # Display a correlation heatmap of the aggregated features.
        scenario = self.scenario_var.get()
        component = self.component_var.get().lower()
        if scenario == "Scenario 1":
            data = load_scenario1_data()
        elif scenario == "Scenario 2":
            data = load_scenario2_data()
        elif scenario == "Scenario 3":
            data = load_scenario3_data()
        else:
            data = None
        if data is None or data.empty:
            messagebox.showerror("Error", "No data loaded for correlation analysis.")
            return
        # Filter data based on system component
        if component == "hydraulic pump":
            filtered_data = data[data['source'].str.lower().str.contains("phydraulique")]
        elif component == "tanks":
            filtered_data = data[data['source'].str.lower().str.contains("tank")]
        elif component == "engines":
            filtered_data = data[data['source'].str.lower().str.contains("driver")]
        elif component == "pumps":
            filtered_data = data[data['source'].str.lower().str.contains("pump") &
                                 ~data['source'].str.lower().str.contains("phydraulique")]
        else:
            filtered_data = data
        if filtered_data.empty:
            messagebox.showerror("Error", "No filtered data for correlation analysis.")
            return

        # Compute the correlation matrix for numeric columns
        corr_matrix = filtered_data.select_dtypes(include=[np.number]).corr()

        # Create a new Toplevel window and make it scrollable
        corr_window = tk.Toplevel(self)
        corr_window.title("Correlation Matrix")

        # Create a canvas with scrollbar
        canvas = tk.Canvas(corr_window)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(corr_window, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create a frame inside the canvas
        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

        # Create the correlation plot inside the frame
        fig, ax = plt.subplots(figsize=(12, 10))
        cax = ax.matshow(corr_matrix, cmap='coolwarm')
        fig.colorbar(cax, ax=ax)

        # Set ticks and labels
        ticks = np.arange(0, len(corr_matrix.columns), 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr_matrix.columns, fontsize=8)

        ax.set_title("Correlation Matrix of Aggregated Sensor Features\n(Mean, Std, Min, Max for each measurement)", pad=40, fontsize=11)
        ax.set_xlabel("Aggregated Features\n(Statistical summaries of sensor data)", fontsize=10, labelpad=15)
        ax.set_ylabel("Aggregated Features\n(Statistical summaries of sensor data)", fontsize=10, labelpad=15)

        annotation_text = (
            "Explanation of Feature Suffixes:\n"
            "- '_mean': Average value over the time window\n"
            "- '_std': Standard deviation (variability)\n"
            "- '_min': Minimum value observed\n"
            "- '_max': Maximum value observed\n\n"
            "Correlation Details:\n"
            "A high positive correlation (close to +1) means two features tend to change together.\n"
            "A negative correlation (close to -1) indicates an inverse relationship.\n"
            "Values near 0 suggest little to no linear relationship."
        )
        fig.tight_layout(rect=[0, 0.15, 1, 1])
        fig.text(0.5, 0.03, annotation_text, ha='center', fontsize=9, wrap=True)

        canvas_fig = FigureCanvasTkAgg(fig, master=frame)
        canvas_fig.draw()
        canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_anomaly_detection(self):
        # Run anomaly detection on the aggregated features using IsolationForest.
        scenario = self.scenario_var.get()
        component = self.component_var.get().lower()
        if scenario == "Scenario 1":
            data = load_scenario1_data()
        elif scenario == "Scenario 2":
            data = load_scenario2_data()
        elif scenario == "Scenario 3":
            data = load_scenario3_data()
        else:
            data = None
        if data is None or data.empty:
            messagebox.showerror("Error", "No data loaded for anomaly detection.")
            return
        if component == "hydraulic pump":
            filtered_data = data[data['source'].str.lower().str.contains("phydraulique")]
        elif component == "tanks":
            filtered_data = data[data['source'].str.lower().str.contains("tank")]
        elif component == "engines":
            filtered_data = data[data['source'].str.lower().str.contains("driver")]
        elif component == "pumps":
            filtered_data = data[data['source'].str.lower().str.contains("pump") &
                                 ~data['source'].str.lower().str.contains("phydraulique")]
        else:
            filtered_data = data
        if filtered_data.empty:
            messagebox.showerror("Error", "No filtered data for anomaly detection.")
            return

        filtered_data = filtered_data.copy()  # Avoid SettingWithCopyWarning
        X = filtered_data.drop(columns=['label', 'source'])
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        preds = iso_forest.fit_predict(X)
        filtered_data['anomaly'] = preds

        print("Total samples:", len(filtered_data))
        print("Normal samples (label 1):", (filtered_data['anomaly'] == 1).sum())
        print("Anomaly samples (label -1):", (filtered_data['anomaly'] == -1).sum())

        numeric_cols = X.columns
        if numeric_cols.empty:
            messagebox.showerror("Error", "No numeric columns for anomaly detection plot.")
            return
        col_to_plot = numeric_cols[0]
        anomaly_window = tk.Toplevel(self)
        anomaly_window.title("Anomaly Detection")
        fig, ax = plt.subplots(figsize=(8, 4))
        normal_data = filtered_data[filtered_data['anomaly'] == 1]
        anomaly_data = filtered_data[filtered_data['anomaly'] == -1]
        
        if normal_data.empty and anomaly_data.empty:
            messagebox.showerror("Error", "Anomaly detection found no data to plot.")
            return

        ax.plot(normal_data.index, normal_data[col_to_plot], 'b.', label='Normal', markersize=6)
        ax.plot(anomaly_data.index, anomaly_data[col_to_plot], 'ro', label='Anomaly', markersize=6)
        ax.set_title(f"Anomaly Detection for {col_to_plot}\n(Component: {component.capitalize()})", fontsize=11)
        ax.set_xlabel("Sample Index (Order of Aggregated Data)", fontsize=10)
        ax.set_ylabel(f"{col_to_plot} Value\n(This sensor reading represents the measured parameter)", fontsize=10)
        ax.legend(loc='best')
        ax.text(0.5, 0.02, "Blue dots indicate normal behavior; red dots indicate anomalies detected by Isolation Forest.", 
                transform=ax.transAxes, fontsize=9, ha='center')
        canvas = FigureCanvasTkAgg(fig, master=anomaly_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = DigitalTwinUI()
    app.mainloop()