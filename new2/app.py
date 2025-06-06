import threading, time, random, hashlib, json, math, os, csv
from collections import deque
from statistics import median
from flask import Flask, jsonify, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# -------------------------
# Simulated Blockchain Part
# -------------------------
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data  # anomaly details, sensor data, and recommendations
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        block_string = f"{self.index}{self.timestamp}{json.dumps(self.data)}{self.previous_hash}"
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
    
    def create_genesis_block(self):
        return Block(0, time.time(), "Genesis Block", "0")
    
    def get_latest_block(self):
        return self.chain[-1]
    
    def add_block(self, new_data):
        latest_block = self.get_latest_block()
        new_block = Block(
            index=latest_block.index + 1,
            timestamp=time.time(),
            data=new_data,
            previous_hash=latest_block.hash
        )
        self.chain.append(new_block)

# Global blockchain instance
blockchain = Blockchain()

# -------------------------
# Robust Statistical Functions
# -------------------------
def compute_mad(data):
    med = median(data)
    deviations = [abs(x - med) for x in data]
    return median(deviations) if deviations else 0

def robust_zscore(x, data):
    med = median(data)
    mad = compute_mad(data)
    if mad == 0:
        return 0
    return 0.6745 * (x - med) / mad

# -------------------------
# Dynamic Feature Engineering Function
# -------------------------
def get_engineered_data():
    try:
        data = pd.read_csv("scenario_data.csv")
    except Exception as e:
        print("Error reading scenario_data.csv:", e)
        return []
    
    if data.empty:
        return []
    
    # Compute derived features dynamically
    data['efficiency_score'] = data['throughput'] / data['energy_consumption'] * 100
    data['avg_temperature'] = (data['avg_T_in'] + data['avg_T_out']) / 2
    data['temp_diff'] = data['avg_T_out'] - data['avg_T_in']
    data['cv_T_in'] = data['std_T_in'] / data['avg_T_in']
    data['cv_T_out'] = data['std_T_out'] / data['avg_T_out']
    
    engineered_data = data.to_dict(orient="records")
    return engineered_data

# -------------------------
# Sensor Data & Enhanced Anomaly Detection
# -------------------------
sensor_data = {}
sensor_history = deque(maxlen=50)

machine_baselines = {
    "machine_1": { 
        "T_in": (25, 5),
        "T_out": (400, 30),
        "RPM": (20000, 1000),
        "Vibration": (0.3, 0.1),
        "PressureRatio": (12, 1)
    },
    "machine_2": { 
        "T_in": (26, 4),
        "T_out": (395, 25),
        "RPM": (19800, 900),
        "Vibration": (0.28, 0.08),
        "PressureRatio": (11.8, 1)
    },
    "machine_3": { 
        "T_in": (24, 6),
        "T_out": (405, 35),
        "RPM": (20100, 1100),
        "Vibration": (0.32, 0.12),
        "PressureRatio": (12.2, 1)
    }
}

machine_sensor_history = { machine: deque(maxlen=50) for machine in machine_baselines }

PR_surge = 10.0
surge_threshold = 15.0
robust_threshold = 3.5

def get_recommendation(reading, anomalies):
    recommendations = []
    if "T_out" in anomalies:
        recommendations.append("High outlet temperature detected. Consider enhancing cooling system.")
    if "SurgeMargin" in anomalies:
        recommendations.append("Low surge margin detected. Consider revising compressor design.")
    if "RPM" in anomalies:
        recommendations.append("High rotor speed detected. Evaluate operating conditions.")
    if "Vibration" in anomalies:
        recommendations.append("Excessive vibration detected. Inspect for blade imbalances.")
    if "PressureRatio" in anomalies:
        recommendations.append("Abnormal pressure ratio detected. Consider modifying design parameters.")
    if "T_in" in anomalies:
        recommendations.append("Inlet temperature deviation detected. Verify ambient conditions.")
    return " ".join(recommendations) if recommendations else "No recommendations; parameters within nominal range."

def generate_sensor_data(machine_id, baseline):
    while True:
        T_in = random.gauss(baseline["T_in"][0], baseline["T_in"][1])
        T_out = random.gauss(baseline["T_out"][0], baseline["T_out"][1])
        RPM = random.gauss(baseline["RPM"][0], baseline["RPM"][1])
        Vibration = random.gauss(baseline["Vibration"][0], baseline["Vibration"][1])
        PressureRatio = random.gauss(baseline["PressureRatio"][0], baseline["PressureRatio"][1])
        surge_margin = ((PressureRatio - PR_surge) / PressureRatio * 100) if PressureRatio else 0
        
        new_reading = {
            "T_in": round(T_in, 2),
            "T_out": round(T_out, 2),
            "RPM": round(RPM, 2),
            "Vibration": round(Vibration, 2),
            "PressureRatio": round(PressureRatio, 2),
            "SurgeMargin": round(surge_margin, 2),
            "timestamp": time.time()
        }
        machine_sensor_history[machine_id].append(new_reading)
        
        anomaly = {}
        if len(machine_sensor_history[machine_id]) >= 5:
            history = machine_sensor_history[machine_id]
            T_in_list = [d["T_in"] for d in history]
            T_out_list = [d["T_out"] for d in history]
            RPM_list = [d["RPM"] for d in history]
            Vib_list = [d["Vibration"] for d in history]
            PR_list = [d["PressureRatio"] for d in history]
            
            rz_T_in = abs(robust_zscore(new_reading["T_in"], T_in_list))
            rz_T_out = abs(robust_zscore(new_reading["T_out"], T_out_list))
            rz_RPM = abs(robust_zscore(new_reading["RPM"], RPM_list))
            rz_Vib = abs(robust_zscore(new_reading["Vibration"], Vib_list))
            rz_PR = abs(robust_zscore(new_reading["PressureRatio"], PR_list))
            
            if rz_T_in > robust_threshold:
                anomaly["T_in"] = round(new_reading["T_in"], 2)
            if rz_T_out > robust_threshold:
                anomaly["T_out"] = round(new_reading["T_out"], 2)
            if rz_RPM > robust_threshold:
                anomaly["RPM"] = round(new_reading["RPM"], 2)
            if rz_Vib > robust_threshold:
                anomaly["Vibration"] = round(new_reading["Vibration"], 2)
            if rz_PR > robust_threshold:
                anomaly["PressureRatio"] = round(new_reading["PressureRatio"], 2)
        
        if surge_margin < surge_threshold:
            anomaly["SurgeMargin"] = round(surge_margin, 2)
        
        if anomaly:
            recommendation = get_recommendation(new_reading, anomaly)
            anomaly_event = {
                "machine_id": machine_id,
                "anomaly": anomaly,
                "sensor_data": new_reading,
                "recommendation": recommendation
            }
            blockchain.add_block(anomaly_event)
            print(f"Anomaly Detected for {machine_id}:", anomaly_event)
        else:
            print(f"Normal Reading for {machine_id}:", new_reading)
        
        time.sleep(5)

for machine_id, baseline in machine_baselines.items():
    t = threading.Thread(target=generate_sensor_data, args=(machine_id, baseline), daemon=True)
    t.start()

# -------------------------
# Periodic CSV Writer
# -------------------------
def periodic_csv_writer(interval, machine_history, filename="scenario_data.csv"):
    from data_collection import aggregate_sensor_data, write_aggregated_data_to_csv
    while True:
        aggregated_data = aggregate_sensor_data(machine_history)
        write_aggregated_data_to_csv(aggregated_data, filename)
        time.sleep(interval)

csv_writer_thread = threading.Thread(
    target=periodic_csv_writer, 
    args=(10, machine_sensor_history, "scenario_data.csv"),
    daemon=True
)
csv_writer_thread.start()

# -------------------------
# Dashboard Template (Digital Twin & Layout Optimizer)
# -------------------------
dashboard_template = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Aerospace Digital Twin Dashboard - Multi-Machine</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      body { padding: 20px; }
      .chart-container { position: relative; height:40vh; width:80vw; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1 class="mb-4">Aerospace Digital Twin & Anomaly Insight Platform (Multi-Machine)</h1>
      <!-- Navigation Buttons -->
      <div class="mb-3">
        <a href="/layout-optimizer" class="btn btn-success">Smart Factory Layout Optimizer</a>
        <a href="/ml" class="btn btn-primary">AI/ML Analysis</a>
      </div>
      <p>This dashboard simulates live sensor data for multiple machines during pre-production testing.
         Anomalies are detected using robust statistics and logged on an immutable blockchain ledger.</p>
      
      <div class="row">
        <div class="col-md-6">
          <h3>Live Sensor Data Chart</h3>
          <canvas id="sensorChart"></canvas>
        </div>
        <div class="col-md-6">
          <h3>Blockchain Ledger</h3>
          <div class="accordion" id="blockchainAccordion">
            {% for block in blockchain %}
            <div class="accordion-item">
              <h2 class="accordion-header" id="heading{{ block.index }}">
                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{{ block.index }}" aria-expanded="false" aria-controls="collapse{{ block.index }}">
                  Block {{ block.index }} - <span class="text-muted">{{ block.timestamp | datetimeformat }}</span>
                </button>
              </h2>
              <div id="collapse{{ block.index }}" class="accordion-collapse collapse" aria-labelledby="heading{{ block.index }}" data-bs-parent="#blockchainAccordion">
                <div class="accordion-body">
                  {% if block.data is string %}
                    <p>{{ block.data }}</p>
                  {% else %}
                    <p><strong>Machine ID:</strong> <span>{{ block.data.machine_id | default("N/A") }}</span></p>
                    <p><strong>Anomaly:</strong> <span class="text-danger">{{ block.data.anomaly | default("None") }}</span></p>
                    <p><strong>Sensor Data:</strong> 
                      T_in: <em>{{ block.data.sensor_data.T_in | default("N/A") }}</em>, 
                      T_out: <em>{{ block.data.sensor_data.T_out | default("N/A") }}</em>, 
                      RPM: <em>{{ block.data.sensor_data.RPM | default("N/A") }}</em>, 
                      Vibration: <em>{{ block.data.sensor_data.Vibration | default("N/A") }}</em>, 
                      Pressure Ratio: <em>{{ block.data.sensor_data.PressureRatio | default("N/A") }}</em>, 
                      Surge Margin: <em>{{ block.data.sensor_data.SurgeMargin | default("N/A") }}</em>
                    </p>
                    {% if block.data.recommendation %}
                      <p><strong>Recommendation:</strong> <span class="text-primary">{{ block.data.recommendation }}</span></p>
                    {% endif %}
                  {% endif %}
                </div>
              </div>
            </div>
            {% endfor %}
          </div>
        </div>
      </div>
      
      <hr>
      <h3>Historical Sensor Data (Last 50 Readings, All Machines)</h3>
      <table class="table table-striped">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Machine ID</th>
            <th>T_in (°C)</th>
            <th>T_out (°C)</th>
            <th>RPM</th>
            <th>Vibration (mm/s)</th>
            <th>Pressure Ratio</th>
            <th>Surge Margin (%)</th>
          </tr>
        </thead>
        <tbody id="historyTableBody">
          {% for reading in sensor_history %}
          <tr>
            <td>{{ reading.timestamp | datetimeformat }}</td>
            <td>{{ reading.machine_id }}</td>
            <td>{{ reading.T_in }}</td>
            <td>{{ reading.T_out }}</td>
            <td>{{ reading.RPM }}</td>
            <td>{{ reading.Vibration }}</td>
            <td>{{ reading.PressureRatio }}</td>
            <td>{{ reading.SurgeMargin }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    
    <!-- Bootstrap Bundle with Popper for accordion functionality -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <!-- JavaScript for live updates -->
    <script>
      function formatTimestamp(ts) {
        var date = new Date(ts * 1000);
        return date.toLocaleTimeString();
      }
      
      var ctx = document.getElementById('sensorChart').getContext('2d');
      var sensorChart = new Chart(ctx, {
          type: 'line',
          data: {
              labels: [],
              datasets: [
                  { label: 'T_in (°C)', borderColor: 'blue', fill: false, data: [] },
                  { label: 'T_out (°C)', borderColor: 'red', fill: false, data: [] },
                  { label: 'RPM', borderColor: 'green', fill: false, data: [] },
                  { label: 'Vibration (mm/s)', borderColor: 'orange', fill: false, data: [] },
                  { label: 'Pressure Ratio', borderColor: 'purple', fill: false, data: [] },
                  { label: 'Surge Margin (%)', borderColor: 'brown', fill: false, data: [] }
              ]
          },
          options: {
              scales: { x: { title: { display: true, text: 'Time' } } },
              responsive: true,
              plugins: { title: { display: true, text: 'Compressor Sensor Data Over Time' } }
          }
      });
      
      async function updateSensorData() {
          const selectedMachine = document.getElementById("machineSelect").value;
          const response = await fetch('/api/history');
          const data = await response.json();
          const filteredData = selectedMachine === "all" ? data : data.filter(item => item.machine_id === selectedMachine);
          const labels = filteredData.map(item => formatTimestamp(item.timestamp));
          sensorChart.data.labels = labels;
          sensorChart.data.datasets[0].data = filteredData.map(item => item.T_in);
          sensorChart.data.datasets[1].data = filteredData.map(item => item.T_out);
          sensorChart.data.datasets[2].data = filteredData.map(item => item.RPM);
          sensorChart.data.datasets[3].data = filteredData.map(item => item.Vibration);
          sensorChart.data.datasets[4].data = filteredData.map(item => item.PressureRatio);
          sensorChart.data.datasets[5].data = filteredData.map(item => item.SurgeMargin);
          sensorChart.update();
      }
      
      async function updateBlockchain() {
          const response = await fetch('/api/blockchain');
          const data = await response.json();
          let accordionHTML = "";
          data.forEach(block => {
              accordionHTML += `
                <div class="accordion-item">
                  <h2 class="accordion-header" id="heading${block.index}">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${block.index}" aria-expanded="false" aria-controls="collapse${block.index}">
                      Block ${block.index} - <span class="text-muted">${new Date(block.timestamp * 1000).toLocaleString()}</span>
                    </button>
                  </h2>
                  <div id="collapse${block.index}" class="accordion-collapse collapse" aria-labelledby="heading${block.index}" data-bs-parent="#blockchainAccordion">
                    <div class="accordion-body">
                      ${typeof block.data === 'string' ? `<p>${block.data}</p>` : `
                        <p><strong>Machine ID:</strong> <span>${block.data.machine_id || "N/A"}</span></p>
                        <p><strong>Anomaly:</strong> <span class="text-danger">${JSON.stringify(block.data.anomaly)}</span></p>
                        <p><strong>Sensor Data:</strong>
                          T_in: <em>${block.data.sensor_data.T_in}</em>,
                          T_out: <em>${block.data.sensor_data.T_out}</em>,
                          RPM: <em>${block.data.sensor_data.RPM}</em>,
                          Vibration: <em>${block.data.sensor_data.Vibration}</em>,
                          Pressure Ratio: <em>${block.data.sensor_data.PressureRatio}</em>,
                          Surge Margin: <em>${block.data.sensor_data.SurgeMargin}</em>
                        </p>
                        ${block.data.recommendation ? `<p><strong>Recommendation:</strong> <span class="text-primary">${block.data.recommendation}</span></p>` : ""}
                      `}
                    </div>
                  </div>
                </div>
              `;
          });
          document.getElementById("blockchainAccordion").innerHTML = accordionHTML;
      }
      
      async function updateHistoryTable() {
          const response = await fetch('/api/history');
          const data = await response.json();
          let tableHTML = "";
          data.forEach(reading => {
              tableHTML += `<tr>
                              <td>${new Date(reading.timestamp * 1000).toLocaleString()}</td>
                              <td>${reading.machine_id}</td>
                              <td>${reading.T_in}</td>
                              <td>${reading.T_out}</td>
                              <td>${reading.RPM}</td>
                              <td>${reading.Vibration}</td>
                              <td>${reading.PressureRatio}</td>
                              <td>${reading.SurgeMargin}</td>
                            </tr>`;
          });
          document.getElementById("historyTableBody").innerHTML = tableHTML;
      }
      
      setInterval(() => {
          updateSensorData();
          updateBlockchain();
          updateHistoryTable();
      }, 10000);
      
      updateSensorData();
      updateBlockchain();
      updateHistoryTable();
    </script>
  </body>
</html>
"""

@app.template_filter('datetimeformat')
def datetimeformat(value):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))

@app.route('/')
def dashboard():
    aggregated_history = []
    for machine_id, history in machine_sensor_history.items():
        for reading in history:
            reading_copy = reading.copy()
            reading_copy["machine_id"] = machine_id
            aggregated_history.append(reading_copy)
    aggregated_history.sort(key=lambda x: x["timestamp"])
    
    chain_data = []
    for block in blockchain.chain:
        chain_data.append({
            "index": block.index,
            "timestamp": block.timestamp,
            "data": block.data,
            "previous_hash": block.previous_hash,
            "hash": block.hash
        })
    return render_template_string(dashboard_template, sensor_history=aggregated_history, blockchain=chain_data)

@app.route('/api/history')
def get_history():
    aggregated_history = []
    for machine_id, history in machine_sensor_history.items():
        for reading in history:
            reading_copy = reading.copy()
            reading_copy["machine_id"] = machine_id
            aggregated_history.append(reading_copy)
    aggregated_history.sort(key=lambda x: x["timestamp"])
    return jsonify(aggregated_history)

@app.route('/api/blockchain')
def get_blockchain():
    chain_data = []
    for block in blockchain.chain:
        chain_data.append({
            "index": block.index,
            "timestamp": block.timestamp,
            "data": block.data,
            "previous_hash": block.previous_hash,
            "hash": block.hash
        })
    return jsonify(chain_data)

@app.route('/aggregate')
def aggregate():
    aggregated_data = aggregate_sensor_data(machine_sensor_history)
    return jsonify(aggregated_data)

# -------------------------
# New AI/ML Analysis UI Page (Dynamic Feature Analysis)
# -------------------------
@app.route('/ml')
def ml_ui():
    return render_template_string(ml_template)

@app.route('/api/engineered')
def engineered_data_api():
    engineered_data = get_engineered_data()
    print("Dynamic engineered data returned:", engineered_data)
    return jsonify(engineered_data)

@app.route('/predict', methods=["POST"])
def predict():
    import pandas as pd
    data_input = request.get_json()
    feature_order = [
        "machine_count",
        "avg_T_in", "std_T_in",
        "avg_T_out", "std_T_out",
        "avg_RPM", "std_RPM",
        "avg_Vibration", "std_Vibration",
        "cycle_time",
        "energy_consumption",
        "estimated_travel_distance"
    ]
    input_df = pd.DataFrame([data_input], columns=feature_order)
    model = joblib.load("throughput_model.pkl")
    predicted = model.predict(input_df)[0]
    return jsonify({"predicted_throughput": round(predicted, 2)})

# -------------------------
# Layout Optimizer & Simulator (Unchanged)
# -------------------------
layout_optimizer_template = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Smart Factory Layout Optimizer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Tailwind CSS -->
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <!-- Konva -->
    <script src="https://cdn.jsdelivr.net/npm/konva@8.3.13/konva.min.js"></script>
    <style>
      #container {
          border: 2px dashed #ccc;
          background-color: #f9f9f9;
      }
    </style>
  </head>
  <body class="bg-gray-100">
    <div class="container mx-auto p-4">
      <h1 class="text-2xl font-bold mb-4">Smart Factory Layout Optimizer</h1>
      <div class="flex flex-col md:flex-row">
        <!-- Sidebar for machine palette -->
        <div class="md:w-1/4 p-4 bg-white shadow rounded mb-4 md:mb-0">
          <h2 class="text-xl mb-2">Machine Palette</h2>
          <div id="palette" class="space-y-2">
            <div class="machine p-2 bg-blue-200 rounded cursor-pointer" data-type="3D Printer">3D Printer</div>
            <div class="machine p-2 bg-green-200 rounded cursor-pointer" data-type="CNC">CNC</div>
            <div class="machine p-2 bg-yellow-200 rounded cursor-pointer" data-type="Assembly Robot">Assembly Robot</div>
            <div class="machine p-2 bg-purple-200 rounded cursor-pointer" data-type="Quality Scanner">Quality Scanner</div>
          </div>
          <button id="optimizeBtn" class="mt-4 px-4 py-2 bg-indigo-600 text-white rounded">Optimize Layout</button>
        </div>
        <!-- Main canvas area -->
        <div class="md:w-3/4 p-4">
          <div id="container" class="mb-4" style="width:800px; height:600px;"></div>
          <div class="flex space-x-4">
            <button id="simulateBtn" class="px-4 py-2 bg-blue-600 text-white rounded">Simulate Layout</button>
            <button id="resetBtn" class="px-4 py-2 bg-red-600 text-white rounded">Reset Layout</button>
          </div>
          <div id="results" class="mt-4 p-4 bg-white shadow rounded"></div>
        </div>
      </div>
    </div>
    
    <script>
      const width = 800;
      const height = 600;
      const stage = new Konva.Stage({
          container: 'container',
          width: width,
          height: height,
          draggable: true
      });
      const gridLayer = new Konva.Layer();
      const machineLayer = new Konva.Layer();
      const pathLayer = new Konva.Layer();
      stage.add(gridLayer);
      stage.add(pathLayer);
      stage.add(machineLayer);
      
      const gridSize = 50;
      for (let i = 0; i < width / gridSize; i++) {
          gridLayer.add(new Konva.Line({
              points: [i * gridSize, 0, i * gridSize, height],
              stroke: '#ddd',
              strokeWidth: 1,
          }));
      }
      for (let j = 0; j < height / gridSize; j++) {
          gridLayer.add(new Konva.Line({
              points: [0, j * gridSize, width, j * gridSize],
              stroke: '#ddd',
              strokeWidth: 1,
          }));
      }
      gridLayer.draw();
      
      let machines = [];
      
      const machineSpecs = {
          "3D Printer": { processing_time: 300, power: 5 },
          "CNC": { processing_time: 180, power: 10 },
          "Assembly Robot": { processing_time: 120, power: 2 },
          "Quality Scanner": { processing_time: 60, power: 1 }
      };
      
      function snapToGrid(value) {
          return Math.round(value / gridSize) * gridSize;
      }
      
      function createMachine(type, x, y) {
          const group = new Konva.Group({
              x: snapToGrid(x),
              y: snapToGrid(y),
              draggable: true,
              name: 'machine'
          });
          const rect = new Konva.Rect({
              width: 80,
              height: 80,
              fill: '#fff',
              stroke: 'black',
              strokeWidth: 1,
          });
          const text = new Konva.Text({
              text: type,
              fontSize: 12,
              width: 80,
              height: 80,
              align: 'center',
              verticalAlign: 'middle'
          });
          group.add(rect);
          group.add(text);
          group.on('dragend', function() {
              group.x(snapToGrid(group.x()));
              group.y(snapToGrid(group.y()));
              checkCollisions();
              machineLayer.draw();
          });
          machineLayer.add(group);
          machineLayer.draw();
          machines.push({ type: type, group: group });
      }
      
      function checkCollisions() {
          let collision = false;
          for (let i = 0; i < machines.length; i++) {
              const a = machines[i].group.getClientRect();
              for (let j = i + 1; j < machines.length; j++) {
                  const b = machines[j].group.getClientRect();
                  if (a.x < b.x + b.width && a.x + a.width > b.x &&
                      a.y < b.y + b.height && a.y + a.height > b.y) {
                          collision = true;
                          machines[i].group.findOne('Rect').stroke('red');
                          machines[j].group.findOne('Rect').stroke('red');
                  } else {
                      machines[i].group.findOne('Rect').stroke('black');
                      machines[j].group.findOne('Rect').stroke('black');
                  }
              }
          }
          machineLayer.draw();
          return collision;
      }
      
      function computeTravelPath(fromGroup, toGroup) {
          const startX = fromGroup.x() + 40;
          const startY = fromGroup.y() + 40;
          const endX = toGroup.x() + 40;
          const endY = toGroup.y() + 40;
          return [startX, startY, endX, endY];
      }
      
      function drawPaths() {
          pathLayer.destroyChildren();
          const order = ["3D Printer", "CNC", "Assembly Robot", "Quality Scanner"];
          let sequence = [];
          for (let type of order) {
              const m = machines.find(m => m.type === type);
              if (m) sequence.push(m);
          }
          if (sequence.length < 2) return;
          for (let i = 0; i < sequence.length - 1; i++) {
              const points = computeTravelPath(sequence[i].group, sequence[i+1].group);
              const arrow = new Konva.Arrow({
                  points: points,
                  pointerLength: 10,
                  pointerWidth: 10,
                  fill: 'orange',
                  stroke: 'orange',
                  strokeWidth: 2,
              });
              pathLayer.add(arrow);
          }
          pathLayer.draw();
      }
      
      function simulateLayout() {
          const order = ["3D Printer", "CNC", "Assembly Robot", "Quality Scanner"];
          let sequence = [];
          for (let type of order) {
              const m = machines.find(m => m.type === type);
              if (m) sequence.push(m);
          }
          if (sequence.length < 2) {
              document.getElementById('results').innerHTML = "<p class='text-red-600'>Please place at least two machines in process order.</p>";
              return;
          }
          let totalProcessing = 0;
          let totalDistance = 0;
          for (let m of sequence) {
              totalProcessing += machineSpecs[m.type].processing_time;
          }
          for (let i = 0; i < sequence.length - 1; i++) {
              const points = computeTravelPath(sequence[i].group, sequence[i+1].group);
              const dx = points[2] - points[0];
              const dy = points[3] - points[1];
              const dist = Math.sqrt(dx*dx + dy*dy);
              totalDistance += dist;
          }
          const transportSpeed = 1;
          const travelTime = totalDistance / transportSpeed;
          const cycleTime = totalProcessing + travelTime;
          const throughput = 3600 / cycleTime;
          let processingEnergy = 0;
          for (let m of sequence) {
              processingEnergy += machineSpecs[m.type].power * (machineSpecs[m.type].processing_time / 3600.0);
          }
          const transportEnergyRate = 0.1;
          const travelEnergy = totalDistance * transportEnergyRate;
          const totalEnergy = processingEnergy + travelEnergy;
          const idealThroughput = 3600 / totalProcessing;
          const idealEnergy = processingEnergy;
          const score = (throughput / idealThroughput) * (idealEnergy / totalEnergy) * 100;
          const collision = checkCollisions();
          let suggestion = "";
          if (collision) {
              suggestion = "Collision detected! Please adjust machine positions.";
          } else {
              suggestion = "Layout looks good!";
          }
          document.getElementById('results').innerHTML = `
              <div class="p-4 bg-gray-100 rounded">
                  <p><strong>Total Processing Time:</strong> ${totalProcessing.toFixed(2)} sec</p>
                  <p><strong>Total Travel Time:</strong> ${travelTime.toFixed(2)} sec</p>
                  <p><strong>Cycle Time:</strong> ${cycleTime.toFixed(2)} sec</p>
                  <p><strong>Throughput:</strong> ${throughput.toFixed(2)} parts/hour</p>
                  <p><strong>Total Energy:</strong> ${totalEnergy.toFixed(2)} kWh per part</p>
                  <p><strong>Suggestion:</strong> ${suggestion}</p>
              </div>
          `;
          drawPaths();
      }
      
      function optimizeLayout() {
          const order = ["3D Printer", "CNC", "Assembly Robot", "Quality Scanner"];
          let sequence = [];
          for (let type of order) {
              const m = machines.find(m => m.type === type);
              if (m) sequence.push(m);
          }
          if (sequence.length < 2) return;
          let maxDist = 0;
          let indexToAdjust = -1;
          for (let i = 0; i < sequence.length - 1; i++) {
              const points = computeTravelPath(sequence[i].group, sequence[i+1].group);
              const dx = points[2] - points[0];
              const dy = points[3] - points[1];
              const dist = Math.sqrt(dx*dx + dy*dy);
              if (dist > maxDist) {
                  maxDist = dist;
                  indexToAdjust = i+1;
              }
          }
          if (indexToAdjust !== -1) {
              const prev = sequence[indexToAdjust - 1].group;
              const curr = sequence[indexToAdjust].group;
              const targetX = (prev.x() + curr.x()) / 2;
              const targetY = (prev.y() + curr.y()) / 2;
              curr.to({
                  x: snapToGrid(targetX),
                  y: snapToGrid(targetY),
                  duration: 0.5
              });
              machineLayer.draw();
              drawPaths();
              alert("AI Suggestion: Adjusted machine position to reduce travel distance.");
          }
      }
      
      document.querySelectorAll("#palette .machine").forEach(item => {
          item.addEventListener("click", () => {
              const type = item.getAttribute("data-type");
              const x = Math.random() * (width - 100);
              const y = Math.random() * (height - 100);
              createMachine(type, x, y);
          });
      });
      
      document.getElementById("simulateBtn").addEventListener("click", simulateLayout);
      document.getElementById("optimizeBtn").addEventListener("click", optimizeLayout);
      document.getElementById("resetBtn").addEventListener("click", () => {
          machines = [];
          machineLayer.destroyChildren();
          pathLayer.destroyChildren();
          machineLayer.draw();
          pathLayer.draw();
          document.getElementById('results').innerHTML = "";
      });
      
      stage.on('wheel', function(e) {
          e.evt.preventDefault();
          var oldScale = stage.scaleX();
          var pointer = stage.getPointerPosition();
          var mousePointTo = {
              x: (pointer.x - stage.x()) / oldScale,
              y: (pointer.y - stage.y()) / oldScale,
          };
          var direction = e.evt.deltaY > 0 ? -1 : 1;
          var factor = 1.05;
          var newScale = direction > 0 ? oldScale * factor : oldScale / factor;
          stage.scale({ x: newScale, y: newScale });
          var newPos = {
              x: pointer.x - mousePointTo.x * newScale,
              y: pointer.y - mousePointTo.y * newScale,
          };
          stage.position(newPos);
          stage.batchDraw();
      });
    </script>
  </body>
</html>
"""

@app.template_filter('datetimeformat')
def datetimeformat(value):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))

# @app.route('/')
# def dashboard():
#     aggregated_history = []
#     for machine_id, history in machine_sensor_history.items():
#         for reading in history:
#             reading_copy = reading.copy()
#             reading_copy["machine_id"] = machine_id
#             aggregated_history.append(reading_copy)
#     aggregated_history.sort(key=lambda x: x["timestamp"])
    
#     chain_data = []
#     for block in blockchain.chain:
#         chain_data.append({
#             "index": block.index,
#             "timestamp": block.timestamp,
#             "data": block.data,
#             "previous_hash": block.previous_hash,
#             "hash": block.hash
#         })
#     return render_template_string(dashboard_template, sensor_history=aggregated_history, blockchain=chain_data)

# @app.route('/api/history')
# def get_history():
#     aggregated_history = []
#     for machine_id, history in machine_sensor_history.items():
#         for reading in history:
#             reading_copy = reading.copy()
#             reading_copy["machine_id"] = machine_id
#             aggregated_history.append(reading_copy)
#     aggregated_history.sort(key=lambda x: x["timestamp"])
#     return jsonify(aggregated_history)

# @app.route('/api/blockchain')
# def get_blockchain():
#     chain_data = []
#     for block in blockchain.chain:
#         chain_data.append({
#             "index": block.index,
#             "timestamp": block.timestamp,
#             "data": block.data,
#             "previous_hash": block.previous_hash,
#             "hash": block.hash
#         })
#     return jsonify(chain_data)

# @app.route('/aggregate')
# def aggregate():
#     aggregated_data = aggregate_sensor_data(machine_sensor_history)
#     return jsonify(aggregated_data)

# -------------------------
# New AI/ML Analysis UI Page (Dynamic Feature Analysis)
# -------------------------
ml_template = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AI/ML Analysis - Factory Performance</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
<div class="container my-4">
  <h1>AI/ML Analysis for Factory Performance</h1>
  <hr>
  
  <h2>Feature Analysis</h2>
  <p>The table below shows the engineered features from aggregated sensor data. This table updates dynamically every 10 seconds.</p>
  <div id="featureTable">
    <!-- Engineered features will be loaded here via AJAX -->
  </div>
  
  <hr>
  
  <h2>Scenario Evaluation</h2>
  <form id="scenarioForm">
    <div class="mb-3">
      <label for="machine_count" class="form-label">Machine Count</label>
      <input type="number" class="form-control" id="machine_count" name="machine_count" value="3" required>
    </div>
    <div class="row">
      <div class="col-md-4">
        <label for="avg_T_in" class="form-label">Average Inlet Temperature (°C)</label>
        <input type="number" step="0.1" class="form-control" id="avg_T_in" name="avg_T_in" value="25.0" required>
      </div>
      <div class="col-md-4">
        <label for="std_T_in" class="form-label">Std. Inlet Temperature</label>
        <input type="number" step="0.1" class="form-control" id="std_T_in" name="std_T_in" value="2.0" required>
      </div>
      <div class="col-md-4">
        <label for="avg_T_out" class="form-label">Average Outlet Temperature (°C)</label>
        <input type="number" step="0.1" class="form-control" id="avg_T_out" name="avg_T_out" value="400.0" required>
      </div>
    </div>
    <div class="row mt-3">
      <div class="col-md-4">
        <label for="std_T_out" class="form-label">Std. Outlet Temperature</label>
        <input type="number" step="0.1" class="form-control" id="std_T_out" name="std_T_out" value="15.0" required>
      </div>
      <div class="col-md-4">
        <label for="avg_RPM" class="form-label">Average RPM</label>
        <input type="number" class="form-control" id="avg_RPM" name="avg_RPM" value="20000" required>
      </div>
      <div class="col-md-4">
        <label for="std_RPM" class="form-label">Std. RPM</label>
        <input type="number" class="form-control" id="std_RPM" name="std_RPM" value="500" required>
      </div>
    </div>
    <div class="row mt-3">
      <div class="col-md-4">
        <label for="avg_Vibration" class="form-label">Average Vibration</label>
        <input type="number" step="0.01" class="form-control" id="avg_Vibration" name="avg_Vibration" value="0.3" required>
      </div>
      <div class="col-md-4">
        <label for="std_Vibration" class="form-label">Std. Vibration</label>
        <input type="number" step="0.01" class="form-control" id="std_Vibration" name="std_Vibration" value="0.05" required>
      </div>
      <div class="col-md-4">
        <label for="cycle_time" class="form-label">Cycle Time (sec)</label>
        <input type="number" step="0.1" class="form-control" id="cycle_time" name="cycle_time" value="50" required>
      </div>
    </div>
    <div class="row mt-3">
      <div class="col-md-6">
        <label for="energy_consumption" class="form-label">Energy Consumption</label>
        <input type="number" step="0.1" class="form-control" id="energy_consumption" name="energy_consumption" value="20" required>
      </div>
      <div class="col-md-6">
        <label for="estimated_travel_distance" class="form-label">Estimated Travel Distance</label>
        <input type="number" step="0.1" class="form-control" id="estimated_travel_distance" name="estimated_travel_distance" value="120" required>
      </div>
    </div>
    <button type="submit" class="btn btn-primary mt-4">Predict Performance</button>
  </form>
  
  <div id="predictionResult" class="mt-4"></div>
</div>

<script>
  // Function to load engineered feature data dynamically
  async function loadFeatureData() {
      const response = await fetch('/api/engineered');
      const data = await response.json();
      let html = '<table class="table table-bordered"><thead><tr>';
      if(data.length > 0) {
          for(let key in data[0]) {
              html += '<th>' + key + '</th>';
          }
      } else {
          html += '<th>No Data Found</th>';
      }
      html += '</tr></thead><tbody>';
      data.forEach(row => {
          html += '<tr>';
          for(let key in row) {
              html += '<td>' + row[key] + '</td>';
          }
          html += '</tr>';
      });
      html += '</tbody></table>';
      document.getElementById('featureTable').innerHTML = html;
  }
  
  // Initial load and periodic refresh every 10 seconds
  loadFeatureData();
  setInterval(loadFeatureData, 10000);
  
  // Handle scenario evaluation form submission
  document.getElementById('scenarioForm').addEventListener('submit', async function(e) {
      e.preventDefault();
      const formData = new FormData(this);
      let payload = {};
      formData.forEach((value, key) => {
          payload[key] = parseFloat(value);
      });
      
      const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
      });
      const result = await response.json();
      document.getElementById('predictionResult').innerHTML = 
          '<div class="alert alert-info"><strong>Predicted Throughput:</strong> ' + result.predicted_throughput + '</div>';
  });
</script>
</body>
</html>
"""

# @app.route('/ml')
# def ml_ui():
#     return render_template_string(ml_template)

# @app.route('/api/engineered')
# def engineered_data_api():
#     engineered_data = get_engineered_data()
#     print("Dynamic engineered data returned:", engineered_data)
#     return jsonify(engineered_data)

# @app.route('/predict', methods=["POST"])
# def predict():
#     import pandas as pd
#     data_input = request.get_json()
#     feature_order = [
#         "machine_count",
#         "avg_T_in", "std_T_in",
#         "avg_T_out", "std_T_out",
#         "avg_RPM", "std_RPM",
#         "avg_Vibration", "std_Vibration",
#         "cycle_time",
#         "energy_consumption",
#         "estimated_travel_distance"
#     ]
#     input_df = pd.DataFrame([data_input], columns=feature_order)
#     model = joblib.load("throughput_model.pkl")
#     predicted = model.predict(input_df)[0]
#     return jsonify({"predicted_throughput": round(predicted, 2)})

# -------------------------
# Layout Optimizer & Simulator (Unchanged)
# -------------------------
layout_optimizer_template = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Smart Factory Layout Optimizer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Tailwind CSS -->
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <!-- Konva -->
    <script src="https://cdn.jsdelivr.net/npm/konva@8.3.13/konva.min.js"></script>
    <style>
      #container {
          border: 2px dashed #ccc;
          background-color: #f9f9f9;
      }
    </style>
  </head>
  <body class="bg-gray-100">
    <div class="container mx-auto p-4">
      <h1 class="text-2xl font-bold mb-4">Smart Factory Layout Optimizer</h1>
      <div class="flex flex-col md:flex-row">
        <!-- Sidebar for machine palette -->
        <div class="md:w-1/4 p-4 bg-white shadow rounded mb-4 md:mb-0">
          <h2 class="text-xl mb-2">Machine Palette</h2>
          <div id="palette" class="space-y-2">
            <div class="machine p-2 bg-blue-200 rounded cursor-pointer" data-type="3D Printer">3D Printer</div>
            <div class="machine p-2 bg-green-200 rounded cursor-pointer" data-type="CNC">CNC</div>
            <div class="machine p-2 bg-yellow-200 rounded cursor-pointer" data-type="Assembly Robot">Assembly Robot</div>
            <div class="machine p-2 bg-purple-200 rounded cursor-pointer" data-type="Quality Scanner">Quality Scanner</div>
          </div>
          <button id="optimizeBtn" class="mt-4 px-4 py-2 bg-indigo-600 text-white rounded">Optimize Layout</button>
        </div>
        <!-- Main canvas area -->
        <div class="md:w-3/4 p-4">
          <div id="container" class="mb-4" style="width:800px; height:600px;"></div>
          <div class="flex space-x-4">
            <button id="simulateBtn" class="px-4 py-2 bg-blue-600 text-white rounded">Simulate Layout</button>
            <button id="resetBtn" class="px-4 py-2 bg-red-600 text-white rounded">Reset Layout</button>
          </div>
          <div id="results" class="mt-4 p-4 bg-white shadow rounded"></div>
        </div>
      </div>
    </div>
    
    <script>
      const width = 800;
      const height = 600;
      const stage = new Konva.Stage({
          container: 'container',
          width: width,
          height: height,
          draggable: true
      });
      const gridLayer = new Konva.Layer();
      const machineLayer = new Konva.Layer();
      const pathLayer = new Konva.Layer();
      stage.add(gridLayer);
      stage.add(pathLayer);
      stage.add(machineLayer);
      
      const gridSize = 50;
      for (let i = 0; i < width / gridSize; i++) {
          gridLayer.add(new Konva.Line({
              points: [i * gridSize, 0, i * gridSize, height],
              stroke: '#ddd',
              strokeWidth: 1,
          }));
      }
      for (let j = 0; j < height / gridSize; j++) {
          gridLayer.add(new Konva.Line({
              points: [0, j * gridSize, width, j * gridSize],
              stroke: '#ddd',
              strokeWidth: 1,
          }));
      }
      gridLayer.draw();
      
      let machines = [];
      
      const machineSpecs = {
          "3D Printer": { processing_time: 300, power: 5 },
          "CNC": { processing_time: 180, power: 10 },
          "Assembly Robot": { processing_time: 120, power: 2 },
          "Quality Scanner": { processing_time: 60, power: 1 }
      };
      
      function snapToGrid(value) {
          return Math.round(value / gridSize) * gridSize;
      }
      
      function createMachine(type, x, y) {
          const group = new Konva.Group({
              x: snapToGrid(x),
              y: snapToGrid(y),
              draggable: true,
              name: 'machine'
          });
          const rect = new Konva.Rect({
              width: 80,
              height: 80,
              fill: '#fff',
              stroke: 'black',
              strokeWidth: 1,
          });
          const text = new Konva.Text({
              text: type,
              fontSize: 12,
              width: 80,
              height: 80,
              align: 'center',
              verticalAlign: 'middle'
          });
          group.add(rect);
          group.add(text);
          group.on('dragend', function() {
              group.x(snapToGrid(group.x()));
              group.y(snapToGrid(group.y()));
              checkCollisions();
              machineLayer.draw();
          });
          machineLayer.add(group);
          machineLayer.draw();
          machines.push({ type: type, group: group });
      }
      
      function checkCollisions() {
          let collision = false;
          for (let i = 0; i < machines.length; i++) {
              const a = machines[i].group.getClientRect();
              for (let j = i + 1; j < machines.length; j++) {
                  const b = machines[j].group.getClientRect();
                  if (a.x < b.x + b.width && a.x + a.width > b.x &&
                      a.y < b.y + b.height && a.y + a.height > b.y) {
                          collision = true;
                          machines[i].group.findOne('Rect').stroke('red');
                          machines[j].group.findOne('Rect').stroke('red');
                  } else {
                      machines[i].group.findOne('Rect').stroke('black');
                      machines[j].group.findOne('Rect').stroke('black');
                  }
              }
          }
          machineLayer.draw();
          return collision;
      }
      
      function computeTravelPath(fromGroup, toGroup) {
          const startX = fromGroup.x() + 40;
          const startY = fromGroup.y() + 40;
          const endX = toGroup.x() + 40;
          const endY = toGroup.y() + 40;
          return [startX, startY, endX, endY];
      }
      
      function drawPaths() {
          pathLayer.destroyChildren();
          const order = ["3D Printer", "CNC", "Assembly Robot", "Quality Scanner"];
          let sequence = [];
          for (let type of order) {
              const m = machines.find(m => m.type === type);
              if (m) sequence.push(m);
          }
          if (sequence.length < 2) return;
          for (let i = 0; i < sequence.length - 1; i++) {
              const points = computeTravelPath(sequence[i].group, sequence[i+1].group);
              const arrow = new Konva.Arrow({
                  points: points,
                  pointerLength: 10,
                  pointerWidth: 10,
                  fill: 'orange',
                  stroke: 'orange',
                  strokeWidth: 2,
              });
              pathLayer.add(arrow);
          }
          pathLayer.draw();
      }
      
      function simulateLayout() {
          const order = ["3D Printer", "CNC", "Assembly Robot", "Quality Scanner"];
          let sequence = [];
          for (let type of order) {
              const m = machines.find(m => m.type === type);
              if (m) sequence.push(m);
          }
          if (sequence.length < 2) {
              document.getElementById('results').innerHTML = "<p class='text-red-600'>Please place at least two machines in process order.</p>";
              return;
          }
          let totalProcessing = 0;
          let totalDistance = 0;
          for (let m of sequence) {
              totalProcessing += machineSpecs[m.type].processing_time;
          }
          for (let i = 0; i < sequence.length - 1; i++) {
              const points = computeTravelPath(sequence[i].group, sequence[i+1].group);
              const dx = points[2] - points[0];
              const dy = points[3] - points[1];
              const dist = Math.sqrt(dx*dx + dy*dy);
              totalDistance += dist;
          }
          const transportSpeed = 1;
          const travelTime = totalDistance / transportSpeed;
          const cycleTime = totalProcessing + travelTime;
          const throughput = 3600 / cycleTime;
          let processingEnergy = 0;
          for (let m of sequence) {
              processingEnergy += machineSpecs[m.type].power * (machineSpecs[m.type].processing_time / 3600.0);
          }
          const transportEnergyRate = 0.1;
          const travelEnergy = totalDistance * transportEnergyRate;
          const totalEnergy = processingEnergy + travelEnergy;
          const idealThroughput = 3600 / totalProcessing;
          const idealEnergy = processingEnergy;
          const score = (throughput / idealThroughput) * (idealEnergy / totalEnergy) * 100;
          const collision = checkCollisions();
          let suggestion = "";
          if (collision) {
              suggestion = "Collision detected! Please adjust machine positions.";
          } else {
              suggestion = "Layout looks good!";
          }
          document.getElementById('results').innerHTML = `
              <div class="p-4 bg-gray-100 rounded">
                  <p><strong>Total Processing Time:</strong> ${totalProcessing.toFixed(2)} sec</p>
                  <p><strong>Total Travel Time:</strong> ${travelTime.toFixed(2)} sec</p>
                  <p><strong>Cycle Time:</strong> ${cycleTime.toFixed(2)} sec</p>
                  <p><strong>Throughput:</strong> ${throughput.toFixed(2)} parts/hour</p>
                  <p><strong>Total Energy:</strong> ${totalEnergy.toFixed(2)} kWh per part</p>
                  <p><strong>Suggestion:</strong> ${suggestion}</p>
              </div>
          `;
          drawPaths();
      }
      
      function optimizeLayout() {
          const order = ["3D Printer", "CNC", "Assembly Robot", "Quality Scanner"];
          let sequence = [];
          for (let type of order) {
              const m = machines.find(m => m.type === type);
              if (m) sequence.push(m);
          }
          if (sequence.length < 2) return;
          let maxDist = 0;
          let indexToAdjust = -1;
          for (let i = 0; i < sequence.length - 1; i++) {
              const points = computeTravelPath(sequence[i].group, sequence[i+1].group);
              const dx = points[2] - points[0];
              const dy = points[3] - points[1];
              const dist = Math.sqrt(dx*dx + dy*dy);
              if (dist > maxDist) {
                  maxDist = dist;
                  indexToAdjust = i+1;
              }
          }
          if (indexToAdjust !== -1) {
              const prev = sequence[indexToAdjust - 1].group;
              const curr = sequence[indexToAdjust].group;
              const targetX = (prev.x() + curr.x()) / 2;
              const targetY = (prev.y() + curr.y()) / 2;
              curr.to({
                  x: snapToGrid(targetX),
                  y: snapToGrid(targetY),
                  duration: 0.5
              });
              machineLayer.draw();
              drawPaths();
              alert("AI Suggestion: Adjusted machine position to reduce travel distance.");
          }
      }
      
      document.querySelectorAll("#palette .machine").forEach(item => {
          item.addEventListener("click", () => {
              const type = item.getAttribute("data-type");
              const x = Math.random() * (width - 100);
              const y = Math.random() * (height - 100);
              createMachine(type, x, y);
          });
      });
      
      document.getElementById("simulateBtn").addEventListener("click", simulateLayout);
      document.getElementById("optimizeBtn").addEventListener("click", optimizeLayout);
      document.getElementById("resetBtn").addEventListener("click", () => {
          machines = [];
          machineLayer.destroyChildren();
          pathLayer.destroyChildren();
          machineLayer.draw();
          pathLayer.draw();
          document.getElementById('results').innerHTML = "";
      });
      
      stage.on('wheel', function(e) {
          e.evt.preventDefault();
          var oldScale = stage.scaleX();
          var pointer = stage.getPointerPosition();
          var mousePointTo = {
              x: (pointer.x - stage.x()) / oldScale,
              y: (pointer.y - stage.y()) / oldScale,
          };
          var direction = e.evt.deltaY > 0 ? -1 : 1;
          var factor = 1.05;
          var newScale = direction > 0 ? oldScale * factor : oldScale / factor;
          stage.scale({ x: newScale, y: newScale });
          var newPos = {
              x: pointer.x - mousePointTo.x * newScale,
              y: pointer.y - mousePointTo.y * newScale,
          };
          stage.position(newPos);
          stage.batchDraw();
      });
    </script>
  </body>
</html>
"""

@app.template_filter('datetimeformat')
def datetimeformat(value):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))

# @app.route('/')
# def dashboard():
#     aggregated_history = []
#     for machine_id, history in machine_sensor_history.items():
#         for reading in history:
#             reading_copy = reading.copy()
#             reading_copy["machine_id"] = machine_id
#             aggregated_history.append(reading_copy)
#     aggregated_history.sort(key=lambda x: x["timestamp"])
    
#     chain_data = []
#     for block in blockchain.chain:
#         chain_data.append({
#             "index": block.index,
#             "timestamp": block.timestamp,
#             "data": block.data,
#             "previous_hash": block.previous_hash,
#             "hash": block.hash
#         })
#     return render_template_string(dashboard_template, sensor_history=aggregated_history, blockchain=chain_data)

# @app.route('/api/history')
# def get_history():
#     aggregated_history = []
#     for machine_id, history in machine_sensor_history.items():
#         for reading in history:
#             reading_copy = reading.copy()
#             reading_copy["machine_id"] = machine_id
#             aggregated_history.append(reading_copy)
#     aggregated_history.sort(key=lambda x: x["timestamp"])
#     return jsonify(aggregated_history)

# @app.route('/api/blockchain')
# def get_blockchain():
#     chain_data = []
#     for block in blockchain.chain:
#         chain_data.append({
#             "index": block.index,
#             "timestamp": block.timestamp,
#             "data": block.data,
#             "previous_hash": block.previous_hash,
#             "hash": block.hash
#         })
#     return jsonify(chain_data)

# @app.route('/aggregate')
# def aggregate():
#     aggregated_data = aggregate_sensor_data(machine_sensor_history)
#     return jsonify(aggregated_data)

# @app.route('/ml')
# def ml_ui():
#     return render_template_string(ml_template)

# @app.route('/api/engineered')
# def engineered_data_api():
#     engineered_data = get_engineered_data()
#     print("Dynamic engineered data returned:", engineered_data)
#     return jsonify(engineered_data)

# @app.route('/predict', methods=["POST"])
# def predict():
#     import pandas as pd
#     data_input = request.get_json()
#     feature_order = [
#         "machine_count",
#         "avg_T_in", "std_T_in",
#         "avg_T_out", "std_T_out",
#         "avg_RPM", "std_RPM",
#         "avg_Vibration", "std_Vibration",
#         "cycle_time",
#         "energy_consumption",
#         "estimated_travel_distance"
#     ]
#     input_df = pd.DataFrame([data_input], columns=feature_order)
#     model = joblib.load("throughput_model.pkl")
#     predicted = model.predict(input_df)[0]
#     return jsonify({"predicted_throughput": round(predicted, 2)})

@app.route('/layout-optimizer')
def layout_optimizer():
    return render_template_string(layout_optimizer_template)

@app.route('/simulateLayout', methods=["POST"])
def simulate_layout_route():
    machines_input = request.get_json()
    result = simulate_layout_api(machines_input)
    return jsonify(result)

def simulate_layout_api(machines_input):
    order = ["3D Printer", "CNC", "Assembly Robot", "Quality Scanner"]
    sequence = []
    for typ in order:
        for m in machines_input:
            if m.get("type") == typ:
                sequence.append(m)
                break
    if not sequence:
        return {"error": "No valid machines in layout."}
    total_processing = sum(machine_specs[m["type"]]["processing_time"] for m in sequence)
    total_distance = 0
    for i in range(len(sequence) - 1):
        dx = sequence[i+1]["x"] - sequence[i]["x"]
        dy = sequence[i+1]["y"] - sequence[i]["y"]
        dist = math.sqrt(dx*dx + dy*dy)
        total_distance += dist
    travel_time = total_distance / transport_speed if transport_speed else 0
    cycle_time = total_processing + travel_time
    throughput = 3600 / cycle_time if cycle_time else 0
    processing_energy = sum(machine_specs[m["type"]]["power"] * (machine_specs[m["type"]]["processing_time"] / 3600.0) for m in sequence)
    travel_energy = total_distance * transport_energy_rate
    total_energy = processing_energy + travel_energy
    ideal_throughput = 3600 / total_processing if total_processing else 0
    ideal_energy = processing_energy
    score = (throughput / ideal_throughput) * (ideal_energy / total_energy) * 100 if ideal_throughput and total_energy else 0
    return {
        "throughput": round(throughput, 2),
        "total_processing_time": total_processing,
        "travel_time": round(travel_time, 2),
        "cycle_time": round(cycle_time, 2),
        "total_energy_kWh": round(total_energy, 2),
        "ideal_throughput": round(ideal_throughput, 2),
        "ideal_energy": round(ideal_energy, 2),
        "layout_score": round(score, 2)
    }

if __name__ == '__main__':
    app.run(debug=True)
