# data_collection.py
import csv
import time
import statistics
from collections import deque
import os



def aggregate_sensor_data(machine_sensor_history):
    """
    Aggregate sensor readings from each machine.
    Computes mean and standard deviation for selected sensor parameters,
    and derives simple performance metrics (cycle_time, throughput, energy consumption).
    """
    aggregated = []
    for machine_id, readings in machine_sensor_history.items():
        if len(readings) == 0:
            continue

        # Compute average and standard deviation for each sensor parameter.
        avg_T_in = statistics.mean([r["T_in"] for r in readings])
        std_T_in = statistics.pstdev([r["T_in"] for r in readings])
        avg_T_out = statistics.mean([r["T_out"] for r in readings])
        std_T_out = statistics.pstdev([r["T_out"] for r in readings])
        avg_RPM = statistics.mean([r["RPM"] for r in readings])
        std_RPM = statistics.pstdev([r["RPM"] for r in readings])
        avg_Vibration = statistics.mean([r["Vibration"] for r in readings])
        std_Vibration = statistics.pstdev([r["Vibration"] for r in readings])

        # For demonstration purposes, we derive a dummy cycle time.
        # (In a real case, cycle time would be computed based on machine processing and travel times.)
        cycle_time = avg_T_in * 0.2 + avg_T_out * 0.01  # Dummy formula
        throughput = 3600 / cycle_time if cycle_time > 0 else 0

        # Dummy energy consumption metric (e.g., based on RPM)
        energy_consumption = avg_RPM * 0.001

        # Dummy estimated travel distance from layout simulation (set a constant or computed value)
        estimated_travel_distance = 100

        aggregated.append({
            "machine_id": machine_id,
            "machine_count": len(machine_sensor_history),
            "avg_T_in": avg_T_in,
            "std_T_in": std_T_in,
            "avg_T_out": avg_T_out,
            "std_T_out": std_T_out,
            "avg_RPM": avg_RPM,
            "std_RPM": std_RPM,
            "avg_Vibration": avg_Vibration,
            "std_Vibration": std_Vibration,
            "cycle_time": cycle_time,
            "throughput": throughput,
            "energy_consumption": energy_consumption,
            "estimated_travel_distance": estimated_travel_distance,
            "timestamp": time.time()
        })
    return aggregated

def write_aggregated_data_to_csv(aggregated_data, filename="scenario_data.csv"):
    # aggregated_data = aggregate_sensor_data()
    fieldnames = [
         "machine_id",
         "machine_count",
         "avg_T_in",
         "std_T_in",
         "avg_T_out",
         "std_T_out",
         "avg_RPM",
         "std_RPM",
         "avg_Vibration",
         "std_Vibration",
         "cycle_time",
         "throughput",
         "energy_consumption",
         "estimated_travel_distance",
         "timestamp"
    ]
    # with open(filename, "w", newline="") as csvfile:
    #      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #      writer.writeheader()
    #      for row in aggregated_data:
    #          writer.writerow(row)
    # print(f"Aggregated data written to {filename}")
    # Check if the file exists
    file_exists = os.path.isfile(filename)
    
    # Open the file in append mode ("a")
    with open(filename, "a", newline="") as csvfile:
         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
         # If file does not exist, write the header
         if not file_exists:
             writer.writeheader()
         for row in aggregated_data:
             writer.writerow(row)
    print(f"Aggregated data appended to {filename}")

if __name__ == "__main__":
    # Wait for some simulation time (e.g., 30 seconds) before aggregating.
    print("Waiting for simulation data to accumulate...")
    time.sleep(30)
    write_aggregated_data_to_csv()
