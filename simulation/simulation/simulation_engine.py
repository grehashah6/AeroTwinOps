import simpy
import random
import numpy as np
from .machine import Machine
from .logging_util import log_event, LOG_FILE, log_df

def run_simulation_until_event_count(target_events=500, num_machines=3, breakdown_prob=0.1):
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create the simulation environment
    env = simpy.Environment()
    repair_resource = simpy.Resource(env, capacity=1)
    
    # Instantiate machines
    machines = [Machine(env, f"Machine_{i+1}", repair_resource, breakdown_prob) 
                for i in range(num_machines)]
    
    # Run simulation until the log contains at least target_events rows
    while len(log_df) < target_events:
        env.step()  # Process the next event
    
    return machines
