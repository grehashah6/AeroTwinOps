from simulation.simulation_engine import run_simulation_until_event_count
import pandas as pd

if __name__ == "__main__":
    # Run simulation until 1000 events are logged
    machines = run_simulation_until_event_count(target_events=500, num_machines=3, breakdown_prob=0.1)
    print("Simulation completed!")
    for machine in machines:
        print(f"{machine.name} produced {machine.processed_parts} parts.")
    
    # Load and display the first few rows of the simulation log
    log_df = pd.read_csv('data/simulation_log.csv')
    print(log_df.head())
    print(f"Total logged events: {len(log_df)}")
