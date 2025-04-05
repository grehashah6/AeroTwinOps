import simpy
import random
from .logging_util import log_event
from .sensor import generate_sensor_data

class Machine:
    def __init__(self, env, name, repair_resource, breakdown_prob=0.1):
        self.env = env
        self.name = name
        self.repair_resource = repair_resource
        self.breakdown_prob = breakdown_prob
        self.processed_parts = 0
        self.action = env.process(self.run())
    
    def run(self):
        while True:
            # Simulate production time for one part (e.g., 5-10 minutes)
            production_time = random.uniform(5, 10)
            yield self.env.timeout(production_time)
            
            # Generate sensor data
            sensor_data = generate_sensor_data()
            
            # Log production event
            log_event(self.env.now, self.name, 'production', sensor_data)
            print(f"{self.env.now:.2f}: {self.name} produced a part with sensor data {sensor_data}")
            
            self.processed_parts += 1
            
            # Random chance for a breakdown
            if random.random() < self.breakdown_prob:
                yield self.env.process(self.breakdown())
    
    def breakdown(self):
        print(f"{self.env.now:.2f}: {self.name} BREAKDOWN!")
        log_event(self.env.now, self.name, 'breakdown')
        
        # Request the repair resource
        with self.repair_resource.request() as request:
            yield request
            repair_time = random.uniform(3, 7)  # Repair takes 3-7 minutes
            yield self.env.timeout(repair_time)
            log_event(self.env.now, self.name, 'repaired')
            print(f"{self.env.now:.2f}: {self.name} repaired and back in production.")
