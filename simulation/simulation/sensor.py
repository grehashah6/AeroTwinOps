import random

def generate_sensor_data():
    temperature = random.uniform(60, 100)  # Example: 60-100°C
    vibration = random.uniform(0.1, 1.0)     # Example: vibration level
    return {'temperature': round(temperature, 2), 'vibration': round(vibration, 2)}
