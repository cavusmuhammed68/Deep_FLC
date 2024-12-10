# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:35:41 2024

@author: cavus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import linprog
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Part 1: Data Loading and Initialization
file_path = r'C:\Users\cavus\Desktop\Margaret - Energies\Residential_25 - 25 - 50.csv'
data = pd.read_csv(file_path)

# Parameters
Battery_Capacity = 4500  # Battery capacity in mAh
Battery_Capacity_kWh = Battery_Capacity * 3.6 / 1000  # Capacity in kWh
SOC_init = 50  # Initial State of Charge (%)
Battery_temp_init = 25  # Initial battery temperature (°C)
SOH_init = 100  # Initial State of Health (%)
Critical_temp = 60  # Critical temperature (°C)
Charging_efficiency = 0.95
Discharging_efficiency = 0.9
Cost_per_kWh = 0.20  # Grid cost in $/kWh
Peak_hour_multiplier = 2  # Cost multiplier during peak hours
Ambient_temperature = 25  # Ambient temperature in °C

# Extract relevant columns and clean data
required_columns = ['P_LD', 'P1 (PV_LD)', 'P4 (BAT_LD)', 'P3 (PV_BAT)']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Missing required columns in dataset: {required_columns}")

data = data[required_columns].fillna(0)  # Replace NaN with 0
data = data.replace([np.inf, -np.inf], 0)  # Replace infinity values with 0

# Extract values
P_LD = data['P_LD'].values
PV_LD = data['P1 (PV_LD)'].values
BAT_LD = data['P4 (BAT_LD)'].values
PV_BAT = data['P3 (PV_BAT)'].values

# Part 2: LSTM and Deep Learning (FCNN) Model for SOC, SOH, and Temperature Prediction

# Add SOC, SOH, and Temp (example initialization)
data['SOC'] = np.linspace(SOC_init, SOC_init - 10, len(data))
data['Temp'] = np.linspace(Battery_temp_init, Battery_temp_init + 5, len(data))
data['SOH'] = np.linspace(SOH_init, SOH_init - 5, len(data))

# Normalize Data for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Prepare Sequences for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-3])  # Use all features except target columns
        y.append(data[i + sequence_length, -3:])  # Targets: SOC, Temp, SOH
    return np.array(X), np.array(y)

sequence_length = 20
X, y = create_sequences(scaled_data, sequence_length)

# Train-Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define LSTM Model
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(sequence_length, X.shape[2]), return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(3)  # Output: SOC, Temp, SOH
])
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train LSTM Model
history = lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Predict with LSTM
predictions = lstm_model.predict(X_test)

# Rescale Predictions and Actuals
scaler_targets = MinMaxScaler()
scaler_targets.min_ = scaler.min_[-3:]
scaler_targets.scale_ = scaler.scale_[-3:]
scaler_targets.data_min_ = scaler.data_min_[-3:]
scaler_targets.data_max_ = scaler.data_max_[-3:]
scaler_targets.feature_range = scaler.feature_range

predictions_rescaled = scaler_targets.inverse_transform(predictions)
actual_values_rescaled = scaler_targets.inverse_transform(y_test)

# Extract Rescaled Predictions and Actual Values
predicted_SOC, predicted_Temp, predicted_SOH = predictions_rescaled[:, 0], predictions_rescaled[:, 1], predictions_rescaled[:, 2]
actual_SOC, actual_Temp, actual_SOH = actual_values_rescaled[:, 0], actual_values_rescaled[:, 1], actual_values_rescaled[:, 2]

# Part 3: Fuzzy Logic System Setup

# Fuzzy Logic System
SoC = ctrl.Antecedent(np.arange(0, 101, 1), 'SoC')
Load = ctrl.Antecedent(np.arange(0, max(P_LD) + 10, 1), 'Load')
Temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'Temperature')
SOH = ctrl.Antecedent(np.arange(0, 101, 1), 'SOH')  # Add SOH as an input
Charging_Priority = ctrl.Consequent(np.arange(0, 101, 1), 'Charging_Priority')
Grid_Priority = ctrl.Consequent(np.arange(0, 101, 1), 'Grid_Priority')

# Membership Functions for SoC
SoC['low'] = fuzz.trimf(SoC.universe, [0, 0, 33])
SoC['medium'] = fuzz.trimf(SoC.universe, [30, 50, 70])
SoC['high'] = fuzz.trimf(SoC.universe, [67, 100, 100])

# Membership Functions for Load
Load['low'] = fuzz.trimf(Load.universe, [0, 0, max(P_LD) / 3])
Load['medium'] = fuzz.trimf(Load.universe, [max(P_LD) / 4, max(P_LD) / 2, 3 * max(P_LD) / 4])
Load['high'] = fuzz.trimf(Load.universe, [2 * max(P_LD) / 3, max(P_LD), max(P_LD)])

# Membership Functions for Temperature
Temperature['low'] = fuzz.trimf(Temperature.universe, [0, 0, 35])
Temperature['medium'] = fuzz.trimf(Temperature.universe, [30, 50, 70])
Temperature['high'] = fuzz.trimf(Temperature.universe, [60, 100, 100])

# Membership Functions for SOH (State of Health)
SOH['low'] = fuzz.trimf(SOH.universe, [0, 0, 50])
SOH['medium'] = fuzz.trimf(SOH.universe, [25, 50, 75])
SOH['high'] = fuzz.trimf(SOH.universe, [50, 100, 100])

# Membership Functions for Charging Priority
Charging_Priority['low'] = fuzz.trimf(Charging_Priority.universe, [0, 0, 50])
Charging_Priority['medium'] = fuzz.trimf(Charging_Priority.universe, [25, 50, 75])
Charging_Priority['high'] = fuzz.trimf(Charging_Priority.universe, [50, 100, 100])

# Membership Functions for Grid Priority
Grid_Priority['low'] = fuzz.trimf(Grid_Priority.universe, [0, 0, 50])
Grid_Priority['medium'] = fuzz.trimf(Grid_Priority.universe, [25, 50, 75])
Grid_Priority['high'] = fuzz.trimf(Grid_Priority.universe, [50, 100, 100])

# Part 4: Defining Fuzzy Rules and System

# Part 4: Defining Fuzzy Rules and System

# Modify fuzzy rules to include SOH as an input
rule1 = ctrl.Rule(SoC['low'] & Load['high'] & SOH['low'], (Charging_Priority['high'], Grid_Priority['high']))
rule2 = ctrl.Rule(SoC['medium'] & Load['medium'] & SOH['medium'], (Charging_Priority['medium'], Grid_Priority['medium']))
rule3 = ctrl.Rule(SoC['high'] | Temperature['high'] | SOH['high'], (Charging_Priority['low'], Grid_Priority['low']))

# Create the Fuzzy Control System
cost_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
cost_sim = ctrl.ControlSystemSimulation(cost_ctrl)

# Create Fuzzy Simulation for charging control
charging_sim = ctrl.ControlSystemSimulation(cost_ctrl)

# Initialize Metrics
SOC_dynamic = [SOC_init]
Temp_dynamic = [Battery_temp_init]
SOH_dynamic = [SOH_init]
Charging_Priority_dynamic = []

# Fuzzy Control with LSTM predictions
for i in range(len(predicted_SOC)):
    # Ensure all inputs are set before running the simulation
    if np.isnan(predicted_SOC[i]) or np.isnan(predicted_Temp[i]) or np.isnan(predicted_SOH[i]):
        print(f"Missing input values at index {i}. Skipping this time step.")
        continue  # Skip this time step if any input is missing

    # Debugging print statement to check input values before running simulation
    print(f"Setting inputs for time step {i}: SOC={predicted_SOC[i]}, Temp={predicted_Temp[i]}, SOH={predicted_SOH[i]}")
    
    # Ensure the 'Load' value is properly set (You should define how to calculate Load here, if it's a predicted value or something else)
    Load_value = P_LD[i]  # Example Load value (can be adjusted if needed)
    
    # Print Load value for debugging
    print(f"Setting Load value for time step {i}: Load={Load_value}")
    
    # Set inputs for fuzzy simulation
    charging_sim.input['SoC'] = predicted_SOC[i]
    charging_sim.input['Temperature'] = predicted_Temp[i]
    charging_sim.input['SOH'] = predicted_SOH[i]  # Now passing SOH to the input
    charging_sim.input['Load'] = Load_value  # Ensure Load is set

    # Perform the fuzzy logic computation
    charging_sim.compute()
    
    # Append the result to the dynamic list
    Charging_Priority_dynamic.append(charging_sim.output['Charging_Priority'])
    

import os
import matplotlib.pyplot as plt

# Set the directory where figures will be saved
output_directory = r'C:\Users\cavus\Desktop\Margaret - Energies\Results'
os.makedirs(output_directory, exist_ok=True)  # Create the folder if it doesn't exist

# Function to save plots with consistent formatting in the specified directory
def save_plot(fig, filename):
    full_path = os.path.join(output_directory, filename)
    fig.savefig(full_path, dpi=600, bbox_inches='tight')

# Visualization of Charging Priority Dynamic Behavior
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Charging_Priority_dynamic, color='green')
ax.set_xlabel('Time Step [hour]', fontsize=16)
ax.set_ylabel('Charging Priority [%]', fontsize=16)
ax.set_title('Dynamic Charging Priority Based on Deep-FLC Predictions', fontsize=16)
ax.legend(fontsize=16)
ax.grid(True)

# Save the plot to the specified directory
save_plot(fig, 'Dynamic_Charging_Priority.png')
plt.show()

print(f"Figure saved as 'Dynamic_Charging_Priority.png' in directory: {output_directory}.")





























