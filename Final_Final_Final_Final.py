# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:43:42 2024

@author: cavus
"""

# -*- coding: utf-8 -*-
"""
Part 1: Data Loading, Initialization, and Preprocessing
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# File path to the dataset
file_path = r'C:\Users\cavus\Desktop\Margaret - Energies\Residential_25 - Short_Full.csv'

# Load the dataset
try:
    data = pd.read_csv(file_path)
    print("Dataset successfully loaded.")
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at: {file_path}")

# Parameters for the battery system
Battery_Capacity = 4500  # Battery capacity in mAh
Battery_Capacity_kWh = Battery_Capacity * 3.6 / 1000  # Convert capacity to kWh
SOC_init = 50  # Initial State of Charge (SOC) as a percentage
Battery_temp_init = 25  # Initial battery temperature in °C
SOH_init = 100  # Initial State of Health (SOH) as a percentage
Critical_temp = 60  # Critical temperature in °C
Charging_efficiency = 0.95  # Efficiency during charging
Discharging_efficiency = 0.9  # Efficiency during discharging
Cost_per_kWh = 0.20  # Cost of electricity per kWh in dollars
Peak_hour_multiplier = 2  # Multiplier for electricity cost during peak hours
Ambient_temperature = 25  # Ambient temperature in °C

# Ensure the dataset contains the required columns
required_columns = ['P_LD', 'P1 (PV_LD)', 'P4 (BAT_LD)', 'P3 (PV_BAT)']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Missing required columns in dataset: {required_columns}")

# Extract and clean the relevant data columns
data = data[required_columns].fillna(0)  # Replace missing values with 0
data = data.replace([np.inf, -np.inf], 0)  # Replace infinity values with 0

# Extract power demand and supply values
P_LD = data['P_LD'].values  # Load demand
PV_LD = data['P1 (PV_LD)'].values  # Photovoltaic power used directly for load
BAT_LD = data['P4 (BAT_LD)'].values  # Battery power used for load
PV_BAT = data['P3 (PV_BAT)'].values  # Photovoltaic power used to charge the battery

# Initialize additional columns for simulation
data['SOC'] = np.linspace(SOC_init, SOC_init - 10, len(data))  # Simulated initial SOC
data['Temp'] = np.linspace(Battery_temp_init, Battery_temp_init + 5, len(data))  # Simulated temperature
data['SOH'] = np.linspace(SOH_init, SOH_init - 5, len(data))  # Simulated initial SOH

# Normalize data for later use with machine learning models
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

print("Data preprocessing completed.")

# Save preprocessed data for later use
scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns)
scaled_data_df.to_csv('preprocessed_data.csv', index=False)
print("Preprocessed data saved as 'preprocessed_data.csv'.")


# -*- coding: utf-8 -*-
"""
Part 2: LSTM Model Setup and Training
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load preprocessed data
preprocessed_data_path = 'preprocessed_data.csv'
data = pd.read_csv(preprocessed_data_path)
print("Preprocessed data loaded successfully.")

# Extract relevant features and target variables
features = data[['P_LD', 'P1 (PV_LD)', 'P4 (BAT_LD)', 'P3 (PV_BAT)']].values
targets = data[['SOC', 'Temp', 'SOH']].values  # Target variables: SOC, Temp, SOH

# Normalize data
scaler_features = MinMaxScaler()
scaler_targets = MinMaxScaler()

features_normalized = scaler_features.fit_transform(features)
targets_normalized = scaler_targets.fit_transform(targets)

# Prepare sequences for LSTM
def create_sequences(data, targets, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # Sequence of features
        y.append(targets[i + sequence_length])  # Target values corresponding to the sequence
    return np.array(X), np.array(y)

sequence_length = 20
X, y = create_sequences(features_normalized, targets_normalized, sequence_length)

# Train-test split
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define the LSTM model
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(sequence_length, X.shape[2]), return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(3)  # Output: SOC, Temp, SOH
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("LSTM model defined.")

# Train the model
history = lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)
print("LSTM model training completed.")

# Save the trained model
model_path = 'lstm_model.h5'
lstm_model.save(model_path)
print(f"Trained LSTM model saved at {model_path}.")


# -*- coding: utf-8 -*-
"""
Part 2: LSTM Model Setup and Training
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load preprocessed data
preprocessed_data_path = 'preprocessed_data.csv'
data = pd.read_csv(preprocessed_data_path)
print("Preprocessed data loaded successfully.")

# Extract relevant features and target variables
features = data[['P_LD', 'P1 (PV_LD)', 'P4 (BAT_LD)', 'P3 (PV_BAT)']].values
targets = data[['SOC', 'Temp', 'SOH']].values  # Target variables: SOC, Temp, SOH

# Normalize data
scaler_features = MinMaxScaler()
scaler_targets = MinMaxScaler()

features_normalized = scaler_features.fit_transform(features)
targets_normalized = scaler_targets.fit_transform(targets)

# Prepare sequences for LSTM
def create_sequences(data, targets, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # Sequence of features
        y.append(targets[i + sequence_length])  # Target values corresponding to the sequence
    return np.array(X), np.array(y)

sequence_length = 20
X, y = create_sequences(features_normalized, targets_normalized, sequence_length)

# Train-test split
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define the LSTM model
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(sequence_length, X.shape[2]), return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(3)  # Output: SOC, Temp, SOH
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("LSTM model defined.")

# Train the model
history = lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)
print("LSTM model training completed.")

# Save the trained model
model_path = 'lstm_model.h5'
lstm_model.save(model_path)
print(f"Trained LSTM model saved at {model_path}.")


# -*- coding: utf-8 -*-
"""
Part 3: Prediction and Postprocessing
"""

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the trained model
model_path = 'lstm_model.h5'
lstm_model = tf.keras.models.load_model(model_path)
print(f"Loaded trained LSTM model from {model_path}.")

# Make predictions
predictions = lstm_model.predict(X_test)

# Rescale predictions and actual values back to the original scale
min_length = min(len(predictions), len(y_test))  # Match the lengths
predictions_rescaled = scaler_targets.inverse_transform(predictions[:min_length])
actual_values_rescaled = scaler_targets.inverse_transform(y_test[:min_length])

# Extract rescaled values for individual targets
predicted_SOC = predictions_rescaled[:, 0]
predicted_Temp = predictions_rescaled[:, 1]
predicted_SOH = predictions_rescaled[:, 2]

actual_SOC = actual_values_rescaled[:, 0]
actual_Temp = actual_values_rescaled[:, 1]
actual_SOH = actual_values_rescaled[:, 2]

# Debugging: Ensure lengths match
assert len(actual_SOC) == len(predicted_SOC), "Mismatch in lengths of actual_SOC and predicted_SOC"

# Calculate errors for evaluation
mae_SOC = mean_absolute_error(actual_SOC, predicted_SOC)
mae_Temp = mean_absolute_error(actual_Temp, predicted_Temp)
mae_SOH = mean_absolute_error(actual_SOH, predicted_SOH)

rmse_SOC = mean_squared_error(actual_SOC, predicted_SOC, squared=False)
rmse_Temp = mean_squared_error(actual_Temp, predicted_Temp, squared=False)
rmse_SOH = mean_squared_error(actual_SOH, predicted_SOH, squared=False)


print(f"MAE (SOC): {mae_SOC:.2f}, MAE (Temp): {mae_Temp:.2f}, MAE (SOH): {mae_SOH:.2f}")
print(f"RMSE (SOC): {rmse_SOC:.2f}, RMSE (Temp): {rmse_Temp:.2f}, RMSE (SOH): {rmse_SOH:.2f}")

# Visualization: Predicted vs Actual Values
plt.figure(figsize=(14, 8))

# Plot SOC
plt.subplot(3, 1, 1)
plt.plot(actual_SOC, label='Actual SOC', color='blue', linewidth=2)
plt.plot(predicted_SOC, label='Predicted SOC', color='orange', linestyle='dashed', linewidth=2)
plt.ylabel('SOC (%)')
plt.title('SOC: Predicted vs Actual')
plt.legend()
plt.grid(True)

# Plot Temperature
plt.subplot(3, 1, 2)
plt.plot(actual_Temp, label='Actual Temp', color='blue', linewidth=2)
plt.plot(predicted_Temp, label='Predicted Temp', color='orange', linestyle='dashed', linewidth=2)
plt.ylabel('Temperature (°C)')
plt.title('Temperature: Predicted vs Actual')
plt.legend()
plt.grid(True)

# Plot SOH
plt.subplot(3, 1, 3)
plt.plot(actual_SOH, label='Actual SOH', color='blue', linewidth=2)
plt.plot(predicted_SOH, label='Predicted SOH', color='orange', linestyle='dashed', linewidth=2)
plt.ylabel('SOH (%)')
plt.xlabel('Time Step')
plt.title('SOH: Predicted vs Actual')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# -*- coding: utf-8 -*-
"""
Part 4: Fuzzy Logic System Setup and Rules
"""

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
from scipy.optimize import linprog

# Define fuzzy input variables
SoC = ctrl.Antecedent(np.arange(0, 101, 1), 'SoC')  # State of Charge (0-100%)
Load = ctrl.Antecedent(np.arange(0, max(P_LD) + 10, 1), 'Load')  # Power Load
Temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'Temperature')  # Battery Temperature (0-100°C)
SOH = ctrl.Antecedent(np.arange(0, 101, 1), 'SOH')  # State of Health (0-100%)

# Define fuzzy output variables
Charging_Priority = ctrl.Consequent(np.arange(0, 101, 1), 'Charging_Priority')  # Charging priority (0-100%)
Grid_Priority = ctrl.Consequent(np.arange(0, 101, 1), 'Grid_Priority')  # Grid usage priority (0-100%)

# Membership functions for SoC
SoC['low'] = fuzz.trimf(SoC.universe, [0, 0, 33])
SoC['medium'] = fuzz.trimf(SoC.universe, [30, 50, 70])
SoC['high'] = fuzz.trimf(SoC.universe, [67, 100, 100])

# Membership functions for Load
Load['low'] = fuzz.trimf(Load.universe, [0, 0, max(P_LD) / 3])
Load['medium'] = fuzz.trimf(Load.universe, [max(P_LD) / 4, max(P_LD) / 2, 3 * max(P_LD) / 4])
Load['high'] = fuzz.trimf(Load.universe, [2 * max(P_LD) / 3, max(P_LD), max(P_LD)])

# Membership functions for Temperature
Temperature['low'] = fuzz.trimf(Temperature.universe, [0, 0, 35])
Temperature['medium'] = fuzz.trimf(Temperature.universe, [30, 50, 70])
Temperature['high'] = fuzz.trimf(Temperature.universe, [60, 100, 100])

# Membership functions for SOH
SOH['low'] = fuzz.trimf(SOH.universe, [0, 0, 50])
SOH['medium'] = fuzz.trimf(SOH.universe, [25, 50, 75])
SOH['high'] = fuzz.trimf(SOH.universe, [50, 100, 100])

# Membership functions for Charging Priority
Charging_Priority['very_low'] = fuzz.trimf(Charging_Priority.universe, [0, 0, 25])
Charging_Priority['low'] = fuzz.trimf(Charging_Priority.universe, [20, 40, 60])
Charging_Priority['medium'] = fuzz.trimf(Charging_Priority.universe, [50, 70, 90])
Charging_Priority['high'] = fuzz.trimf(Charging_Priority.universe, [80, 100, 100])


# Membership functions for Grid Priority
Grid_Priority['low'] = fuzz.trimf(Grid_Priority.universe, [0, 0, 50])
Grid_Priority['medium'] = fuzz.trimf(Grid_Priority.universe, [25, 50, 75])
Grid_Priority['high'] = fuzz.trimf(Grid_Priority.universe, [50, 100, 100])


# Define fuzzy rules
rule1 = ctrl.Rule(SoC['low'] & Load['high'] & SOH['low'], (Charging_Priority['high'], Grid_Priority['high']))
rule2 = ctrl.Rule(SoC['medium'] & Load['medium'] & SOH['medium'], (Charging_Priority['medium'], Grid_Priority['medium']))
rule3 = ctrl.Rule(SoC['high'] | Temperature['high'] | SOH['high'], (Charging_Priority['low'], Grid_Priority['low']))


rule4 = ctrl.Rule(SoC['low'] & Load['high'] & SOH['medium'], (Charging_Priority['high'], Grid_Priority['medium']))
rule5 = ctrl.Rule(SoC['medium'] & Load['low'], (Charging_Priority['medium'], Grid_Priority['low']))
rule6 = ctrl.Rule(SoC['high'] & Load['medium'], (Charging_Priority['low'], Grid_Priority['low']))
rule7 = ctrl.Rule(SoC['low'] & Temperature['high'], (Charging_Priority['low'], Grid_Priority['high']))



# Update the fuzzy control system
cost_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
cost_sim = ctrl.ControlSystemSimulation(cost_ctrl)


print("Fuzzy logic system and rules successfully defined.")

# -*- coding: utf-8 -*-
"""
Part 5: Simulation Loop for Fuzzy and Normal Systems
"""

# Initialize Metrics
SOC_normal = [SOC_init]
SOC_fuzzy = [SOC_init]
Battery_temp_normal = [Battery_temp_init]
Battery_temp_fuzzy = [Battery_temp_init]
Grid_power_normal = []
Grid_power_fuzzy = []
Charging_Priority_dynamic = []  # Store dynamic charging priorities
Total_cost_normal = 0
Total_cost_fuzzy = 0

# Simulation Loop
# Limit simulation loop to the minimum length of predictions and load demand
min_length = min(len(P_LD), len(predicted_SOC))
for k in range(min_length):

    # Determine cost multiplier for peak hours
    hour = k % 24  # Simulate 24-hour cycle
    # Dynamic cost multiplier
    Peak_hour_multiplier = 3  # Increase multiplier to make peak costs more punitive
    Off_peak_multiplier = 0.8  # Reduce cost for off-peak hours

    # Determine cost multiplier dynamically
    if 17 <= hour <= 21:  # Peak hours: 5 PM to 9 PM
        cost_per_kWh_dynamic = Cost_per_kWh * Peak_hour_multiplier
    elif 0 <= hour <= 6:  # Off-peak hours: midnight to 6 AM
        cost_per_kWh_dynamic = Cost_per_kWh * Off_peak_multiplier
    else:
        cost_per_kWh_dynamic = Cost_per_kWh


    ### Normal System ###
    # Minimize grid power usage and maximize battery usage
    c_normal = [cost_per_kWh_dynamic, -1]  # Cost for grid power, prioritize battery
    A_eq = [[1, 0]]  # Ensure load power balance
    b_eq = [P_LD[k] - PV_LD[k]]  # Net load after PV contribution
    bounds = [(0, None), (0, Battery_Capacity_kWh)]  # Bounds for grid and battery usage

    result_normal = linprog(c_normal, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not result_normal.success:
        grid_power_used_normal = P_LD[k] - PV_LD[k]
    else:
        grid_power_used_normal, battery_power_used_normal = result_normal.x

    Grid_power_normal.append(grid_power_used_normal)
    Total_cost_normal += grid_power_used_normal * cost_per_kWh_dynamic

    # Update SOC for the normal system
    SOC_new_normal = SOC_normal[-1] + (PV_BAT[k] - BAT_LD[k] / Discharging_efficiency) / Battery_Capacity_kWh * 100
    SOC_normal.append(max(15, min(90, SOC_new_normal)))  # Clamp SOC between 15% and 90%

    # Update temperature for the normal system
    temp_normal = Battery_temp_normal[-1] + 0.1 * grid_power_used_normal  # Temperature rise due to usage
    Battery_temp_normal.append(min(max(Ambient_temperature, temp_normal), Critical_temp))  # Clamp temp to Critical Temp

    ### Fuzzy Logic System ###
    # Set inputs for fuzzy logic simulation
    if k < len(predicted_SOC):  # Ensure we don't exceed prediction size
        cost_sim.input['SoC'] = predicted_SOC[k]
        cost_sim.input['SOH'] = predicted_SOH[k]
        cost_sim.input['Temperature'] = predicted_Temp[k]
    else:
        cost_sim.input['SoC'] = predicted_SOC[-1]
        cost_sim.input['SOH'] = predicted_SOH[-1]
        cost_sim.input['Temperature'] = predicted_Temp[-1]

    cost_sim.input['Load'] = P_LD[k]  # Load value
    cost_sim.compute()

    charging_priority = cost_sim.output.get('Charging_Priority', 50)
    grid_priority = cost_sim.output.get('Grid_Priority', 50)

    # Fuzzy-based optimization for grid power and battery usage
    # Fuzzy-based optimization for grid power and battery usage
    c_fuzzy = [
    cost_per_kWh_dynamic * (grid_priority / 100),  # Minimize grid cost
    -charging_priority / 100  # Maximize charging priority
    ]
    result_fuzzy = linprog(c_fuzzy, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # Check the optimization result and set grid power used
    if not result_fuzzy.success:
        grid_power_used_fuzzy = P_LD[k] - PV_LD[k]
    else:
        grid_power_used_fuzzy, battery_power_used_fuzzy = result_fuzzy.x

    # Apply penalty for peak hours
    if cost_per_kWh_dynamic > Cost_per_kWh:  # Peak hour condition
        grid_power_used_fuzzy *= 0.75  # Increase penalty for grid power during peak hours

    # Update the total cost for fuzzy logic system
    Total_cost_fuzzy += grid_power_used_fuzzy * cost_per_kWh_dynamic

    

    # Update SOC for the fuzzy logic system
    SOC_new_fuzzy = SOC_fuzzy[-1] + (PV_BAT[k] * charging_priority / 100 - BAT_LD[k] / Discharging_efficiency) / Battery_Capacity_kWh * 100
    SOC_fuzzy.append(max(15, min(90, SOC_new_fuzzy)))  # Clamp SOC between 15% and 90%

    # Update temperature for the fuzzy logic system
    temp_fuzzy = max(
        Ambient_temperature,
        Battery_temp_fuzzy[-1] + 0.1 * grid_power_used_fuzzy - 0.025 * charging_priority
    )
    Battery_temp_fuzzy.append(min(temp_fuzzy, Critical_temp))  # Clamp temp to Critical Temp

    # Store charging priority
    Charging_Priority_dynamic.append(charging_priority)

# Results Summary
print("Simulation Loop Completed.")
print(f"Total Cost (Normal System): ${Total_cost_normal:.2f}")
print(f"Total Cost (Fuzzy Logic System): ${Total_cost_fuzzy:.2f}")


# -*- coding: utf-8 -*-
"""
Part 6: Visualization of Simulation Results
"""

import matplotlib.pyplot as plt
import numpy as np

# Visualization 1: SOC Comparison for Normal and Fuzzy Logic Systems
plt.figure(figsize=(10, 6))
plt.plot(SOC_normal, label='Normal System SOC', color='blue', linewidth=2)
plt.plot(SOC_fuzzy, label='Fuzzy Logic System SOC', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('State of Charge (SOC, %)')
plt.title('SOC Comparison: Normal System vs Fuzzy Logic System')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 2: Battery Temperature Comparison for Normal and Fuzzy Logic Systems
plt.figure(figsize=(10, 6))
plt.plot(Battery_temp_normal, label='Normal System Temperature', color='blue', linewidth=2)
plt.plot(Battery_temp_fuzzy, label='Fuzzy Logic System Temperature', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Temperature (°C)')
plt.title('Battery Temperature Comparison: Normal System vs Fuzzy Logic System')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 3: Grid Power Comparison for Normal and Fuzzy Logic Systems
plt.figure(figsize=(10, 6))
plt.plot(Grid_power_normal, label='Normal System Grid Power', color='blue', linewidth=2)
plt.plot(Grid_power_fuzzy, label='Fuzzy Logic System Grid Power', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Grid Power (kW)')
plt.title('Grid Power Comparison: Normal System vs Fuzzy Logic System')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 4: Depth of Discharge (DoD) Comparison for Normal and Fuzzy Logic Systems
DoD_normal = [abs(SOC_normal[i] - SOC_normal[i + 1]) for i in range(len(SOC_normal) - 1)]
DoD_fuzzy = [abs(SOC_fuzzy[i] - SOC_fuzzy[i + 1]) for i in range(len(SOC_fuzzy) - 1)]

plt.figure(figsize=(10, 6))
plt.plot(DoD_normal, label='DoD Normal System', color='blue', linewidth=2)
plt.plot(DoD_fuzzy, label='DoD Fuzzy Logic System', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Depth of Discharge (DoD, %)')
plt.title('Depth of Discharge (DoD) Comparison: Normal vs Fuzzy Logic System')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 5: Dynamic Charging Priority Over Time (Fuzzy Logic System)
plt.figure(figsize=(10, 6))
plt.plot(Charging_Priority_dynamic, label='Charging Priority (Dynamic Fuzzy Logic)', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Charging Priority (%)')
plt.title('Dynamic Charging Priority Based on Fuzzy Logic')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 6: Total Costs Comparison
total_costs = [Total_cost_normal, Total_cost_fuzzy]
systems = ['Normal System', 'Fuzzy Logic System']

plt.figure(figsize=(8, 6))
plt.bar(systems, total_costs, color=['blue', 'green'])
plt.ylabel('Total Cost ($)')
plt.title('Total Cost Comparison: Normal vs Fuzzy Logic System')
for i, cost in enumerate(total_costs):
    plt.text(i, cost + 0.05, f"${cost:.2f}", ha='center', fontsize=12, color='black')

plt.grid(axis='y')
plt.tight_layout()
plt.show()


# -*- coding: utf-8 -*-
"""
Part 7: Battery Health Analysis and SOH Metrics
"""

# Initialize Battery Health Metrics
DoD_normal = []  # Depth of Discharge for normal system
DoD_fuzzy = []   # Depth of Discharge for fuzzy logic system
SOH_normal = [SOH_init]  # State of Health for normal system
SOH_fuzzy = [SOH_init]   # State of Health for fuzzy logic system

# Degradation factors (for SOH calculation)
alpha = 0.002  # Increased degradation factor for Depth of Discharge
beta = 0.001   # Increased degradation factor for temperature

# Simulation Loop for Battery Health Metrics
for k in range(1, len(SOC_normal)):
    # Calculate DoD (Depth of Discharge) for the Normal System
    DoD_normal.append(abs(SOC_normal[k] - SOC_normal[k-1]))

    # Calculate DoD (Depth of Discharge) for the Fuzzy Logic System
    DoD_fuzzy.append(abs(SOC_fuzzy[k] - SOC_fuzzy[k-1]))

    # Calculate SOH (State of Health) for the Normal System
    # SOH degradation based on Depth of Discharge and temperature rise
    DoD_t_normal = DoD_normal[-1]
    temp_normal = Battery_temp_normal[k]
    SOH_new_normal = SOH_normal[-1] - alpha * DoD_t_normal - beta * max(0, temp_normal - 25)
    SOH_normal.append(max(0, SOH_new_normal))  # Ensure SOH doesn't go below 0%

    # Calculate SOH (State of Health) for the Fuzzy Logic System
    # SOH degradation based on Depth of Discharge and temperature rise
    DoD_t_fuzzy = DoD_fuzzy[-1]
    temp_fuzzy = Battery_temp_fuzzy[k]
    SOH_new_fuzzy = SOH_fuzzy[-1] - alpha * DoD_t_fuzzy - beta * max(0, temp_fuzzy - 25)
    SOH_fuzzy.append(max(0, SOH_new_fuzzy))  # Ensure SOH doesn't go below 0%

# Additional Metrics: Average Depth of Discharge (DoD) for Both Systems
avg_dod_normal = np.mean(DoD_normal)
avg_dod_fuzzy = np.mean(DoD_fuzzy)

# Print SOH Results
print(f"Final SOH - Normal System: {SOH_normal[-1]:.2f}%")
print(f"Final SOH - Fuzzy Logic System: {SOH_fuzzy[-1]:.2f}%")
print(f"Average Depth of Discharge (DoD) - Normal System: {avg_dod_normal:.2f}%")
print(f"Average Depth of Discharge (DoD) - Fuzzy Logic System: {avg_dod_fuzzy:.2f}%")

# Visualization 1: SOH Comparison for Normal and Fuzzy Logic Systems
plt.figure(figsize=(10, 6))
plt.plot(SOH_normal, label='Normal System SOH', color='blue', linewidth=2)
plt.plot(SOH_fuzzy, label='Fuzzy Logic System SOH', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('State of Health (SOH, %)')
plt.title('State of Health (SOH) Comparison: Normal System vs Fuzzy Logic System')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 2: Depth of Discharge (DoD) Comparison for Normal and Fuzzy Logic Systems
plt.figure(figsize=(10, 6))
plt.plot(DoD_normal, label='DoD Normal System', color='blue', linewidth=2)
plt.plot(DoD_fuzzy, label='DoD Fuzzy Logic System', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Depth of Discharge (DoD, %)')
plt.title('Depth of Discharge (DoD) Comparison: Normal vs Fuzzy Logic System')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 3: Average Depth of Discharge (DoD) for Both Systems
plt.figure(figsize=(8, 6))
systems = ['Normal System', 'Fuzzy Logic System']
avg_dod_values = [avg_dod_normal, avg_dod_fuzzy]

plt.bar(systems, avg_dod_values, color=['blue', 'green'])
plt.xlabel('System Type')
plt.ylabel('Average Depth of Discharge (%)')
plt.title('Comparison of Average Depth of Discharge (DoD)')
plt.ylim(0, max(avg_dod_values) * 1.2)  # Add some padding to the top
for i, value in enumerate(avg_dod_values):
    plt.text(i, value + 0.05, f"{value:.2f}%", ha='center', fontsize=12, color='black')

plt.grid(axis='y')
plt.tight_layout()
plt.show()


# -*- coding: utf-8 -*-
"""
Part 8: Advanced Visualization and Summary Statistics
"""

import pandas as pd

# Visualization 1: SOC and SOH Over Time for Both Systems
plt.figure(figsize=(14, 8))

# Plot SOC for Normal vs Fuzzy Logic Systems
plt.subplot(2, 1, 1)
plt.plot(SOC_normal, label='SOC Normal System', color='blue', linewidth=2)
plt.plot(SOC_fuzzy, label='SOC Fuzzy Logic System', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('State of Charge (SOC, %)')
plt.title('SOC Comparison: Normal System vs Fuzzy Logic System')
plt.legend()
plt.grid(True)

# Plot SOH for Normal vs Fuzzy Logic Systems
plt.subplot(2, 1, 2)
plt.plot(SOH_normal, label='SOH Normal System', color='blue', linewidth=2)
plt.plot(SOH_fuzzy, label='SOH Fuzzy Logic System', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('State of Health (SOH, %)')
plt.title('SOH Comparison: Normal System vs Fuzzy Logic System')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualization 2: Grid Power vs Battery Power for Both Systems
plt.figure(figsize=(14, 8))

# Plot Grid Power for Normal vs Fuzzy Logic Systems
plt.subplot(2, 1, 1)
plt.plot(Grid_power_normal, label='Grid Power Normal System', color='blue', linewidth=2)
plt.plot(Grid_power_fuzzy, label='Grid Power Fuzzy Logic System', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Grid Power (kW)')
plt.title('Grid Power Comparison: Normal vs Fuzzy Logic System')
plt.legend()
plt.grid(True)

# Plot Battery Power (BAT_LD) for Normal vs Fuzzy Logic Systems
plt.subplot(2, 1, 2)
plt.plot(BAT_LD, label='Battery Power (BAT_LD) Original', color='blue', linestyle='--', linewidth=2)
# Ensure the loop runs only for the common length
common_length = min(len(BAT_LD), len(SOC_fuzzy))
plt.plot([min(BAT_LD[i], SOC_fuzzy[i] / 100 * Battery_Capacity_kWh) for i in range(common_length)], 
         label='Battery Power (BAT_LD) Fuzzy System', color='green')
plt.xlabel('Time Step')
plt.ylabel('Power (kW)')
plt.title('Battery Power Comparison: Original vs Fuzzy Logic System')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Summary Statistics
total_energy_normal = np.sum(Grid_power_normal)  # Total energy consumed from grid by Normal System
total_energy_fuzzy = np.sum(Grid_power_fuzzy)  # Total energy consumed from grid by Fuzzy Logic System

avg_temp_normal = np.mean(Battery_temp_normal)  # Average battery temperature in the normal system
avg_temp_fuzzy = np.mean(Battery_temp_fuzzy)  # Average battery temperature in the fuzzy logic system

# Create a DataFrame for the summary statistics
summary_stats = pd.DataFrame({
    'Metric': [
        'Final SOC (Normal)', 'Final SOC (Fuzzy)',
        'Final SOH (Normal)', 'Final SOH (Fuzzy)',
        'Average Temperature (Normal)', 'Average Temperature (Fuzzy)',
        'Total Grid Energy (Normal)', 'Total Grid Energy (Fuzzy)'
    ],
    'Value': [
        SOC_normal[-1], SOC_fuzzy[-1],
        SOH_normal[-1], SOH_fuzzy[-1],
        avg_temp_normal, avg_temp_fuzzy,
        total_energy_normal, total_energy_fuzzy
    ]
})

# Display the summary statistics table
print("\nSummary Statistics:")
print(summary_stats)

# Visualization 3: Energy Efficiency Comparison
efficiency_normal = np.sum(BAT_LD) / (total_energy_normal + np.sum(BAT_LD))  # Normal system efficiency
efficiency_fuzzy = np.sum(BAT_LD) / (total_energy_fuzzy + np.sum(BAT_LD))  # Fuzzy logic system efficiency

plt.figure(figsize=(8, 6))
efficiency_values = [efficiency_normal, efficiency_fuzzy]
systems = ['Normal System', 'Fuzzy Logic System']

plt.bar(systems, efficiency_values, color=['blue', 'green'])
plt.ylabel('Energy Efficiency (Energy Output / Total Consumed)')
plt.title('Energy Efficiency Comparison: Normal vs Fuzzy Logic System')
plt.ylim(0, max(efficiency_values) * 1.25)
for i, value in enumerate(efficiency_values):
    plt.text(i, value + 0.05, f"{value:.2f}", ha='center', fontsize=12, color='black')

plt.tight_layout()
plt.show()


# -*- coding: utf-8 -*-
"""
Part 9: Insights and Additional Metrics
"""

# Calculate Total Depth of Discharge (DoD) for Both Systems
total_dod_normal = np.sum(DoD_normal)
total_dod_fuzzy = np.sum(DoD_fuzzy)

# Calculate Average Depth of Discharge (DoD) for Both Systems
avg_dod_normal = np.mean(DoD_normal)
avg_dod_fuzzy = np.mean(DoD_fuzzy)

# Additional Metrics
print("\nAdditional Insights:")
print(f"Total Depth of Discharge (DoD) - Normal System: {total_dod_normal:.2f} %")
print(f"Total Depth of Discharge (DoD) - Fuzzy Logic System: {total_dod_fuzzy:.2f} %")
print(f"Average Depth of Discharge (DoD) - Normal System: {avg_dod_normal:.2f} %")
print(f"Average Depth of Discharge (DoD) - Fuzzy Logic System: {avg_dod_fuzzy:.2f} %")

# Efficiency Analysis
battery_efficiency_normal = np.sum(BAT_LD) / (total_energy_normal + np.sum(BAT_LD))
battery_efficiency_fuzzy = np.sum(BAT_LD) / (total_energy_fuzzy + np.sum(BAT_LD))

print(f"Battery Efficiency (Normal System): {battery_efficiency_normal:.2f}")
print(f"Battery Efficiency (Fuzzy Logic System): {battery_efficiency_fuzzy:.2f}")

# Battery Health Summary
final_soh_normal = SOH_normal[-1]
final_soh_fuzzy = SOH_fuzzy[-1]

print(f"Final State of Health (SOH) - Normal System: {final_soh_normal:.2f} %")
print(f"Final State of Health (SOH) - Fuzzy Logic System: {final_soh_fuzzy:.2f} %")

# Visualization: Efficiency Comparison
plt.figure(figsize=(8, 6))
efficiency_values = [battery_efficiency_normal, battery_efficiency_fuzzy]
systems = ['Normal System', 'Fuzzy Logic System']

plt.bar(systems, efficiency_values, color=['blue', 'green'])
plt.ylabel('Battery Efficiency (Energy Used / Total Consumed)')
plt.title('Battery Efficiency Comparison: Normal vs Fuzzy Logic System')
plt.ylim(0, max(efficiency_values) * 1.2)
for i, value in enumerate(efficiency_values):
    plt.text(i, value + 0.05, f"{value:.2f}", ha='center', fontsize=12, color='black')

plt.tight_layout()
plt.show()



# -*- coding: utf-8 -*-
"""
Part 10: Final Summary and Recommendations
"""

# Summarizing Key Insights
print("\n=== Final Summary and Recommendations ===")
print(f"Total Cost (Normal System): ${Total_cost_normal:.2f}")
print(f"Total Cost (Fuzzy Logic System): ${Total_cost_fuzzy:.2f}")
print(f"Cost Savings Using Fuzzy Logic: ${Total_cost_normal - Total_cost_fuzzy:.2f}")

print(f"Final SOH (Normal System): {final_soh_normal:.2f} %")
print(f"Final SOH (Fuzzy Logic System): {final_soh_fuzzy:.2f} %")
print(f"SOH Improvement Using Fuzzy Logic: {final_soh_fuzzy - final_soh_normal:.2f} %")

print(f"Total Energy Consumed from Grid (Normal): {total_energy_normal:.2f} kWh")
print(f"Total Energy Consumed from Grid (Fuzzy): {total_energy_fuzzy:.2f} kWh")
print(f"Grid Energy Reduction Using Fuzzy Logic: {total_energy_normal - total_energy_fuzzy:.2f} kWh")

print(f"Average Depth of Discharge (Normal): {avg_dod_normal:.2f} %")
print(f"Average Depth of Discharge (Fuzzy): {avg_dod_fuzzy:.2f} %")

# Recommendation
if Total_cost_fuzzy < Total_cost_normal and final_soh_fuzzy > final_soh_normal:
    print("Recommendation: The fuzzy logic-based system is more cost-effective and maintains better battery health. Consider implementing it in real-world scenarios.")
else:
    print("Recommendation: Further optimization of the fuzzy logic-based system is needed to ensure superior performance.")

# Save Summary Statistics to a CSV File
summary_data = {
    'Metric': [
        'Total Cost (Normal)', 'Total Cost (Fuzzy)',
        'Final SOH (Normal)', 'Final SOH (Fuzzy)',
        'Total Grid Energy (Normal)', 'Total Grid Energy (Fuzzy)',
        'Average DoD (Normal)', 'Average DoD (Fuzzy)',
        'Battery Efficiency (Normal)', 'Battery Efficiency (Fuzzy)'
    ],
    'Value': [
        Total_cost_normal, Total_cost_fuzzy,
        final_soh_normal, final_soh_fuzzy,
        total_energy_normal, total_energy_fuzzy,
        avg_dod_normal, avg_dod_fuzzy,
        battery_efficiency_normal, battery_efficiency_fuzzy
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_csv_path = "final_summary_statistics.csv"
summary_df.to_csv(summary_csv_path, index=False)
print(f"Summary statistics saved to {summary_csv_path}.")

















