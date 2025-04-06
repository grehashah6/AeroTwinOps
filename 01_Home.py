import streamlit as st

st.set_page_config(page_title="Home | AeroTwinOps", layout="wide")

st.title("🚀 Welcome to AeroTwinOps")

st.markdown("---")

# Overview
st.header("📌 Project Overview")
st.markdown("""
**AeroTwinOps** is a Digital Twin Simulation and Predictive Analytics system designed to improve real-time monitoring, fault detection, and predictive maintenance in industrial systems.

Using live sensor data from multiple components (Hydraulic Pumps, Engines, Tanks, etc.), the platform applies advanced ML techniques like:
- 📊 Feature aggregation via sliding windows
- 🌲 Random Forest for classification
- 🚨 Isolation Forest for anomaly detection
- 📉 Correlation and time-series analysis

All of this is visualized through a user-friendly interface — enabling smarter decisions and minimizing downtime.
""")

# About
# st.header("🧠 About the Project")
# st.markdown("""
# This project was developed as part of the **Honeywell Hackathon 2025** and brings together core principles of:

# - Digital Twin Simulations
# - Predictive Maintenance
# - Scalable Machine Learning
# - Data Visualization

# The system is modular, extensible, and built using:
# - **Python**
# - **Streamlit** (for UI)
# - **scikit-learn** (for ML)
# - **pandas, matplotlib, numpy** (for data wrangling + plots)
# """)

# Team
# st.header("🤝 Meet the Team")
# st.markdown("""
# **👩‍💻 Greha Shah**  
# *MS CS @ ASU | Data & AI Enthusiast*  
# - Role: Lead Developer & Data Scientist  
# - Worked on: System architecture, data preprocessing pipeline, ML model integration, Streamlit UI

# ---

# > 💡 _“Our goal was to create an intuitive and powerful dashboard that empowers engineers to make informed decisions at a glance.”_
# """)

# Footer
st.markdown("---")
st.markdown("© 2025 AeroTwinOps Team. Built with ❤️ using Streamlit.")
