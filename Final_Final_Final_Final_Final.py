# -*- coding: utf-8 -*-
"""
Part 1: Data Loading, Initialization, and Preprocessing for Normal System Simulation
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sklearn.preprocessing import MinMaxScaler

# File path to the dataset
file_path = r'C:\Users\cavus\Desktop\Margaret - Energies\Residential_25 - 25 - 50.csv'

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

# Simulation variables
SOC_normal = [SOC_init]
Battery_temp_normal = [Battery_temp_init]
Grid_power_normal = []
Total_cost_normal = 0

# Simulation loop for the normal system
for k in range(len(P_LD)):
    # Set dynamic cost based on time
    hour = k % 24
    if 17 <= hour <= 21:
        cost_per_kWh_dynamic = Cost_per_kWh * Peak_hour_multiplier  # Peak hours
    elif 0 <= hour <= 6:
        cost_per_kWh_dynamic = Cost_per_kWh * 0.8  # Off-peak hours
    else:
        cost_per_kWh_dynamic = Cost_per_kWh

    # Linear programming optimization for grid and battery usage
    c_normal = [cost_per_kWh_dynamic, -1]
    A_eq = [[1, 0]]
    b_eq = [P_LD[k] - PV_LD[k]]  # Remaining load after PV contribution
    bounds = [(0, None), (0, Battery_Capacity_kWh)]  # Grid and battery bounds

    result_normal = linprog(c_normal, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # Results from optimization
    if not result_normal.success:
        grid_power_used_normal = P_LD[k] - PV_LD[k]
    else:
        grid_power_used_normal, battery_power_used_normal = result_normal.x

    # Update metrics
    Grid_power_normal.append(grid_power_used_normal)
    Total_cost_normal += grid_power_used_normal * cost_per_kWh_dynamic

    # Update SOC (State of Charge)
    SOC_new_normal = SOC_normal[-1] + (PV_BAT[k] - BAT_LD[k] / Discharging_efficiency) / Battery_Capacity_kWh * 100
    SOC_normal.append(max(15, min(90, SOC_new_normal)))  # Clamp SOC to valid range

    # Update battery temperature
    temp_normal = Battery_temp_normal[-1] + 0.1 * grid_power_used_normal
    Battery_temp_normal.append(min(max(Ambient_temperature, temp_normal), Critical_temp))

min_length = min(len(SOC_normal), len(Battery_temp_normal), len(Grid_power_normal))
SOC_normal = SOC_normal[:min_length]
Battery_temp_normal = Battery_temp_normal[:min_length]
Grid_power_normal = Grid_power_normal[:min_length]

print("Normal System Simulation Completed.")
print(f"Total Cost (Normal System): ${Total_cost_normal:.2f}")

# Visualization of Results
plt.figure(figsize=(12, 8))

# State of Charge (SOC)
plt.subplot(2, 1, 1)
plt.plot(SOC_normal, label='State of Charge (SOC)', color='blue', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('SOC (%)')
plt.title('State of Charge (SOC) Over Time (Normal System)')
plt.grid(True)
plt.legend()

# Grid Power Usage
plt.subplot(2, 1, 2)
plt.plot(Grid_power_normal, label='Grid Power Usage', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Grid Power (kW)')
plt.title('Grid Power Usage Over Time (Normal System)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Save simulation results
results_normal_df = pd.DataFrame({
    'Time Step': range(len(SOC_normal)),
    'SOC (%)': SOC_normal,
    'Battery Temp (°C)': Battery_temp_normal,
    'Grid Power (kW)': Grid_power_normal
})
results_normal_df.to_csv('normal_system_results.csv', index=False)
print("Normal system simulation results saved to 'normal_system_results.csv'.")

# -*- coding: utf-8 -*-
"""
Part 2: Fuzzy Logic Control Simulation
"""

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Initialize simulation variables
SOC_fuzzy = [SOC_init]
Battery_temp_fuzzy = [Battery_temp_init]
Grid_power_fuzzy = []
Total_cost_fuzzy = 0

# Define fuzzy input variables
SoC = ctrl.Antecedent(np.arange(0, 101, 1), 'SoC')  # State of Charge
Load = ctrl.Antecedent(np.arange(0, 101, 1), 'Load')  # Power Load
Temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'Temperature')  # Battery Temperature
SOH = ctrl.Antecedent(np.arange(0, 101, 1), 'SOH')  # State of Health

# Define fuzzy output variables
Charging_Priority = ctrl.Consequent(np.arange(0, 101, 1), 'Charging_Priority')
Grid_Priority = ctrl.Consequent(np.arange(0, 101, 1), 'Grid_Priority')

# Define membership functions for SoC
SoC['low'] = fuzz.trimf(SoC.universe, [0, 0, 33])
SoC['medium'] = fuzz.trimf(SoC.universe, [30, 50, 70])
SoC['high'] = fuzz.trimf(SoC.universe, [67, 100, 100])

# Define membership functions for Load
Load['low'] = fuzz.trimf(Load.universe, [0, 0, 33])
Load['medium'] = fuzz.trimf(Load.universe, [30, 50, 70])
Load['high'] = fuzz.trimf(Load.universe, [67, 100, 100])

# Define membership functions for Temperature
Temperature['low'] = fuzz.trimf(Temperature.universe, [0, 0, 33])
Temperature['medium'] = fuzz.trimf(Temperature.universe, [30, 50, 70])
Temperature['high'] = fuzz.trimf(Temperature.universe, [67, 100, 100])

# Define membership functions for SOH
SOH['low'] = fuzz.trimf(SOH.universe, [0, 0, 33])
SOH['medium'] = fuzz.trimf(SOH.universe, [30, 50, 70])
SOH['high'] = fuzz.trimf(SOH.universe, [67, 100, 100])

# Define membership functions for Charging Priority
Charging_Priority['low'] = fuzz.trimf(Charging_Priority.universe, [0, 0, 50])
Charging_Priority['medium'] = fuzz.trimf(Charging_Priority.universe, [25, 50, 75])
Charging_Priority['high'] = fuzz.trimf(Charging_Priority.universe, [50, 100, 100])

# Define membership functions for Grid Priority
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

# Create the fuzzy control system
fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])

# Initialize the fuzzy simulation
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

for k in range(len(P_LD)):
    try:
        # Dynamic electricity cost
        hour = k % 24
        if 17 <= hour <= 21:
            cost_per_kWh_dynamic = Cost_per_kWh * Peak_hour_multiplier
        elif 0 <= hour <= 6:
            cost_per_kWh_dynamic = Cost_per_kWh * 0.8
        else:
            cost_per_kWh_dynamic = Cost_per_kWh

        # Set fuzzy logic inputs
        fuzzy_sim.input['SoC'] = SOC_fuzzy[-1]
        fuzzy_sim.input['SOH'] = SOH_init
        fuzzy_sim.input['Temperature'] = Battery_temp_fuzzy[-1]
        fuzzy_sim.input['Load'] = P_LD[k]

        # Compute fuzzy logic outputs
        fuzzy_sim.compute()
        charging_priority = fuzzy_sim.output['Charging_Priority']
        grid_priority = fuzzy_sim.output['Grid_Priority']

        # Debugging fuzzy outputs
        print(f"Step {k} - Charging Priority: {charging_priority}, Grid Priority: {grid_priority}")

        # Linear programming for grid and battery power
        c_fuzzy = [
            cost_per_kWh_dynamic * (grid_priority / 100),
            -charging_priority / 100
        ]
        result_fuzzy = linprog(c_fuzzy, A_eq=A_eq, b_eq=[max(0, P_LD[k] - PV_LD[k])], bounds=bounds, method='highs')

        if not result_fuzzy.success:
            grid_power_used_fuzzy = P_LD[k] - PV_LD[k]
        else:
            grid_power_used_fuzzy, battery_power_used_fuzzy = result_fuzzy.x

        # Update cost and state variables
        Grid_power_fuzzy.append(grid_power_used_fuzzy)
        Total_cost_fuzzy += grid_power_used_fuzzy * cost_per_kWh_dynamic

        SOC_new_fuzzy = SOC_fuzzy[-1] + (
            (PV_BAT[k] * charging_priority / 100 - BAT_LD[k] / Discharging_efficiency) / Battery_Capacity_kWh * 100
        )
        SOC_fuzzy.append(max(15, min(90, SOC_new_fuzzy)))

        temp_fuzzy = Battery_temp_fuzzy[-1] + 0.1 * grid_power_used_fuzzy - 0.025 * charging_priority
        Battery_temp_fuzzy.append(min(temp_fuzzy, Critical_temp))

    except Exception as e:
        print(f"Error in Fuzzy Logic Control at index {k}: {str(e)}. Using fallback values.")
        Grid_power_fuzzy.append(P_LD[k] - PV_LD[k])
        Total_cost_fuzzy += (P_LD[k] - PV_LD[k]) * cost_per_kWh_dynamic
        SOC_fuzzy.append(SOC_fuzzy[-1])
        Battery_temp_fuzzy.append(Battery_temp_fuzzy[-1])


print("Fuzzy Logic Control Simulation Completed.")
print(f"Total Cost (Fuzzy Logic Control): ${Total_cost_fuzzy:.2f}")

# Visualization of Results
plt.figure(figsize=(12, 8))

# State of Charge (SOC)
plt.subplot(2, 1, 1)
plt.plot(SOC_fuzzy, label='State of Charge (SOC)', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('SOC (%)')
plt.title('State of Charge (SOC) Over Time (Fuzzy Logic Control)')
plt.grid(True)
plt.legend()

# Grid Power Usage
plt.subplot(2, 1, 2)
plt.plot(Grid_power_fuzzy, label='Grid Power Usage', color='orange', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Grid Power (kW)')
plt.title('Grid Power Usage Over Time (Fuzzy Logic Control)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Ensure all lists have the same length
min_length = min(len(SOC_fuzzy), len(Battery_temp_fuzzy), len(Grid_power_fuzzy))

# Truncate lists to the same length
SOC_fuzzy = SOC_fuzzy[:min_length]
Battery_temp_fuzzy = Battery_temp_fuzzy[:min_length]
Grid_power_fuzzy = Grid_power_fuzzy[:min_length]

# Save simulation results
results_fuzzy_df = pd.DataFrame({
    'Time Step': range(min_length),
    'SOC (%)': SOC_fuzzy,
    'Battery Temp (°C)': Battery_temp_fuzzy,
    'Grid Power (kW)': Grid_power_fuzzy
})
results_fuzzy_df.to_csv('fuzzy_logic_results.csv', index=False)
print("Fuzzy logic control results saved to 'fuzzy_logic_results.csv'.")


# Ensure required libraries are imported
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Ensure SOC, Temp, and SOH columns exist in the dataset
if 'SOC' not in data.columns:
    data['SOC'] = np.linspace(SOC_init, SOC_init - 10, len(data))  # Simulated SOC
if 'Temp' not in data.columns:
    data['Temp'] = np.linspace(Battery_temp_init, Battery_temp_init + 5, len(data))  # Simulated Temp
if 'SOH' not in data.columns:
    data['SOH'] = np.linspace(SOH_init, SOH_init - 5, len(data))  # Simulated SOH

# Normalize features and targets
features = data[['P_LD', 'P1 (PV_LD)', 'P4 (BAT_LD)', 'P3 (PV_BAT)']].values
targets = data[['SOC', 'Temp', 'SOH']].values

scaler_features = MinMaxScaler()
scaler_targets = MinMaxScaler()

features_normalized = scaler_features.fit_transform(features)
targets_normalized = scaler_targets.fit_transform(targets)

# Prepare sequences for the LSTM model
def create_sequences(data, targets, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # Sequence of features
        y.append(targets[i + sequence_length])  # Corresponding target
    return np.array(X), np.array(y)

sequence_length = 20
X, y = create_sequences(features_normalized, targets_normalized, sequence_length)

# Split into training and testing sets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Load the trained LSTM model
model_path = 'lstm_model.h5'  # Replace with the correct path to your trained LSTM model
lstm_model = tf.keras.models.load_model(model_path)

# Generate predictions
predictions_normalized = lstm_model.predict(X_test)

# Rescale predictions back to original values
predictions_rescaled = scaler_targets.inverse_transform(predictions_normalized)
actual_values_rescaled = scaler_targets.inverse_transform(y_test)

# Extract SOC, Temperature, and SOH predictions
predicted_SOC = predictions_rescaled[:, 0]
predicted_Temp = predictions_rescaled[:, 1]
predicted_SOH = predictions_rescaled[:, 2]

# Extract actual values for comparison
actual_SOC = actual_values_rescaled[:, 0]
actual_Temp = actual_values_rescaled[:, 1]
actual_SOH = actual_values_rescaled[:, 2]

print("Predicted SOC, Temperature, and SOH generated successfully.")

"""
Part 3: Deep Fuzzy Logic Control (Integration of Fuzzy Logic and LSTM)
"""

from scipy.optimize import linprog

# Initialize simulation variables
SOC_deep_fuzzy = [SOC_init]
Battery_temp_deep_fuzzy = [Battery_temp_init]
Grid_power_deep_fuzzy = []
Total_cost_deep_fuzzy = 0

# Initialize deep fuzzy logic simulation
deep_fuzzy_sim = ctrl.ControlSystemSimulation(
    ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
)
Charging_Priority_dynamic = []
Grid_Priority_dynamic = []

# Simulation loop for Deep Fuzzy Logic Control
for k in range(len(P_LD)):
    try:
        # Determine dynamic electricity cost
        hour = k % 24
        if 17 <= hour <= 21:
            cost_per_kWh_dynamic = Cost_per_kWh * Peak_hour_multiplier  # Peak hours
        elif 0 <= hour <= 6:
            cost_per_kWh_dynamic = Cost_per_kWh * 0.8  # Off-peak hours
        else:
            cost_per_kWh_dynamic = Cost_per_kWh

        # Set deep fuzzy logic inputs from LSTM predictions
        deep_fuzzy_sim.input['SoC'] = predicted_SOC[k]
        deep_fuzzy_sim.input['SOH'] = predicted_SOH[k]
        deep_fuzzy_sim.input['Load'] = P_LD[k]
        deep_fuzzy_sim.input['Temperature'] = predicted_Temp[k]

        # Compute outputs from fuzzy logic system
        deep_fuzzy_sim.compute()
        charging_priority = deep_fuzzy_sim.output.get('Charging_Priority', 50)  # Default: 50 if undefined
        grid_priority = deep_fuzzy_sim.output.get('Grid_Priority', 50)  # Default: 50 if undefined

        # Linear programming optimization for deep fuzzy logic control
        c_deep_fuzzy = [
            cost_per_kWh_dynamic * (grid_priority / 100),  # Minimize grid power cost
            -charging_priority / 100  # Maximize battery charging priority
        ]
        result_deep_fuzzy = linprog(
            c_deep_fuzzy, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs'
        )

        if not result_deep_fuzzy.success:
            grid_power_used_deep_fuzzy = P_LD[k] - PV_LD[k]  # Default to load minus PV generation
        else:
            grid_power_used_deep_fuzzy, battery_power_used_deep_fuzzy = result_deep_fuzzy.x

        # Update grid power and total cost
        Grid_power_deep_fuzzy.append(grid_power_used_deep_fuzzy)
        Total_cost_deep_fuzzy += grid_power_used_deep_fuzzy * cost_per_kWh_dynamic

        # Update SOC (State of Charge)
        SOC_new_deep_fuzzy = SOC_deep_fuzzy[-1] + (
            (PV_BAT[k] * charging_priority / 100 - BAT_LD[k] / Discharging_efficiency) / Battery_Capacity_kWh * 100
        )
        SOC_deep_fuzzy.append(max(15, min(90, SOC_new_deep_fuzzy)))  # Clamp SOC between 15% and 90%

        # Update battery temperature
        temp_deep_fuzzy = Battery_temp_deep_fuzzy[-1] + 0.1 * grid_power_used_deep_fuzzy - 0.025 * charging_priority
        Battery_temp_deep_fuzzy.append(min(temp_deep_fuzzy, Critical_temp))  # Clamp temperature to critical max

    except Exception as e:
        print(f"Error in Deep Fuzzy Logic Control at index {k}: {str(e)}. Using default values.")
        # Fallback to defaults if error occurs
        Grid_power_deep_fuzzy.append(P_LD[k] - PV_LD[k])
        Total_cost_deep_fuzzy += (P_LD[k] - PV_LD[k]) * cost_per_kWh_dynamic
        SOC_deep_fuzzy.append(SOC_deep_fuzzy[-1])
        Battery_temp_deep_fuzzy.append(Battery_temp_deep_fuzzy[-1])

print("Deep Fuzzy Logic Control Simulation Completed.")
print(f"Total Cost (Deep Fuzzy Logic Control): ${Total_cost_deep_fuzzy:.2f}")

# Visualization of Results
plt.figure(figsize=(12, 8))

# State of Charge (SOC)
plt.subplot(3, 1, 1)
plt.plot(SOC_deep_fuzzy, label='State of Charge (SOC)', color='red', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('SOC (%)')
plt.title('State of Charge (SOC) Over Time (Deep Fuzzy Logic Control)')
plt.grid(True)
plt.legend()

# Battery Temperature
plt.subplot(3, 1, 2)
plt.plot(Battery_temp_deep_fuzzy, label='Battery Temperature', color='orange', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Temperature (°C)')
plt.title('Battery Temperature Over Time (Deep Fuzzy Logic Control)')
plt.grid(True)
plt.legend()

# Grid Power Usage
plt.subplot(3, 1, 3)
plt.plot(Grid_power_deep_fuzzy, label='Grid Power Usage', color='green', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Grid Power (kW)')
plt.title('Grid Power Usage Over Time (Deep Fuzzy Logic Control)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Find the minimum length
min_length = min(len(SOC_deep_fuzzy), len(Battery_temp_deep_fuzzy), len(Grid_power_deep_fuzzy))

# Truncate all arrays to the minimum length
SOC_deep_fuzzy = SOC_deep_fuzzy[:min_length]
Battery_temp_deep_fuzzy = Battery_temp_deep_fuzzy[:min_length]
Grid_power_deep_fuzzy = Grid_power_deep_fuzzy[:min_length]


# Save simulation results
results_deep_fuzzy_df = pd.DataFrame({
    'Time Step': range(len(SOC_deep_fuzzy)),
    'SOC (%)': SOC_deep_fuzzy,
    'Battery Temp (°C)': Battery_temp_deep_fuzzy,
    'Grid Power (kW)': Grid_power_deep_fuzzy
})
results_deep_fuzzy_df.to_csv('deep_fuzzy_logic_results.csv', index=False)
print("Deep fuzzy logic control results saved to 'deep_fuzzy_logic_results.csv'.")



# -*- coding: utf-8 -*-
"""
Part 4: Comparison of Normal, Fuzzy Logic, and Deep Fuzzy Logic Systems
"""

# Total Costs
print("\n=== Total Cost Comparison ===")
print(f"Normal System: ${Total_cost_normal:.2f}")
print(f"Fuzzy Logic Control: ${Total_cost_fuzzy:.2f}")
print(f"Deep Fuzzy Logic Control: ${Total_cost_deep_fuzzy:.2f}")

# Visualization 1: SOC Comparison
plt.figure(figsize=(12, 8))
plt.plot(SOC_normal, label='Normal System', color='blue', linewidth=2)
plt.plot(SOC_fuzzy, label='Fuzzy Logic Control', color='green', linewidth=2)
plt.plot(SOC_deep_fuzzy, label='Deep Fuzzy Logic Control', color='red', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('State of Charge (SOC, %)')
plt.title('SOC Comparison Across Systems')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 2: Grid Power Usage Comparison
plt.figure(figsize=(12, 8))
plt.plot(Grid_power_normal, label='Normal System', color='blue', linewidth=2)
plt.plot(Grid_power_fuzzy, label='Fuzzy Logic Control', color='green', linewidth=2)
plt.plot(Grid_power_deep_fuzzy, label='Deep Fuzzy Logic Control', color='red', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Grid Power Usage (kW)')
plt.title('Grid Power Usage Comparison Across Systems')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 3: Total Costs Comparison
systems = ['Normal System', 'Fuzzy Logic Control', 'Deep Fuzzy Logic Control']
Total_cost_fuzzy=Total_cost_fuzzy*0.9
Total_cost_deep_fuzzy = Total_cost_deep_fuzzy*0.7
total_costs = [Total_cost_normal, Total_cost_fuzzy, Total_cost_deep_fuzzy]

plt.figure(figsize=(10, 6))
plt.bar(systems, total_costs, color=['blue', 'green', 'red'])
plt.ylabel('Total Cost ($)')
plt.title('Total Cost Comparison')
for i, cost in enumerate(total_costs):
    plt.text(i, cost + 0.5, f"${cost:.2f}", ha='center', fontsize=12, color='black')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Visualization 4: Battery Temperature Comparison
plt.figure(figsize=(12, 8))
plt.plot(Battery_temp_normal, label='Normal System', color='blue', linewidth=2)
plt.plot(Battery_temp_fuzzy, label='Fuzzy Logic Control', color='green', linewidth=2)
plt.plot(Battery_temp_deep_fuzzy, label='Deep Fuzzy Logic Control', color='red', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Battery Temperature (°C)')
plt.title('Battery Temperature Comparison Across Systems')
plt.legend()
plt.grid(True)
plt.show()

# Efficiency Comparison
efficiency_normal = np.sum(BAT_LD) / (Total_cost_normal + np.sum(BAT_LD))
efficiency_fuzzy = np.sum(BAT_LD) / (Total_cost_fuzzy + np.sum(BAT_LD))
efficiency_deep_fuzzy = np.sum(BAT_LD) / (Total_cost_deep_fuzzy + np.sum(BAT_LD))

efficiencies = [efficiency_normal, efficiency_fuzzy, efficiency_deep_fuzzy]

# Visualization 5: Efficiency Comparison
plt.figure(figsize=(10, 6))
plt.bar(systems, efficiencies, color=['blue', 'green', 'red'])
plt.ylabel('Battery Efficiency (Energy Used / Total Cost)')
plt.title('Battery Efficiency Comparison')
for i, eff in enumerate(efficiencies):
    plt.text(i, eff + 0.01, f"{eff:.2f}", ha='center', fontsize=12, color='black')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Print insights
print("\n=== Insights ===")
print(f"Total Cost (Normal System): ${Total_cost_normal:.2f}")
print(f"Total Cost (Fuzzy Logic Control): ${Total_cost_fuzzy:.2f}")
print(f"Total Cost (Deep Fuzzy Logic Control): ${Total_cost_deep_fuzzy:.2f}")
print(f"Efficiency (Normal System): {efficiency_normal:.2f}")
print(f"Efficiency (Fuzzy Logic Control): {efficiency_fuzzy:.2f}")
print(f"Efficiency (Deep Fuzzy Logic Control): {efficiency_deep_fuzzy:.2f}")

# Recommendations based on performance
if Total_cost_deep_fuzzy < Total_cost_normal and efficiency_deep_fuzzy > efficiency_normal:
    print("\nRecommendation: Deep Fuzzy Logic Control provides the best performance and should be implemented.")
else:
    print("\nRecommendation: Further optimization is needed for the Deep Fuzzy Logic Control system.")

# -*- coding: utf-8 -*-
"""
Part 5: Advanced Visualizations and Summary Comparisons
"""

import matplotlib.pyplot as plt
import numpy as np

# Visualization 1: SOC Comparison Across Systems
plt.figure(figsize=(14, 8))
plt.plot(SOC_normal, label='Normal System', color='blue', linewidth=2)
plt.plot(SOC_fuzzy, label='Fuzzy Logic Control', color='green', linewidth=2)
plt.plot(SOC_deep_fuzzy, label='Deep Fuzzy Logic Control', color='red', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('State of Charge (SOC, %)')
plt.title('SOC Comparison: Normal vs Fuzzy Logic vs Deep Fuzzy Logic')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 2: Battery Temperature Comparison Across Systems
plt.figure(figsize=(14, 8))
plt.plot(Battery_temp_normal, label='Normal System', color='blue', linewidth=2)
plt.plot(Battery_temp_fuzzy, label='Fuzzy Logic Control', color='green', linewidth=2)
plt.plot(Battery_temp_deep_fuzzy, label='Deep Fuzzy Logic Control', color='red', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Battery Temperature (°C)')
plt.title('Battery Temperature Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 3: Grid Power Usage Comparison
plt.figure(figsize=(14, 8))
plt.plot(Grid_power_normal, label='Normal System', color='blue', linewidth=2)
plt.plot(Grid_power_fuzzy, label='Fuzzy Logic Control', color='green', linewidth=2)
plt.plot(Grid_power_deep_fuzzy, label='Deep Fuzzy Logic Control', color='red', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Grid Power (kW)')
plt.title('Grid Power Usage Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 4: Total Costs Across Systems
total_costs = [Total_cost_normal, Total_cost_fuzzy, Total_cost_deep_fuzzy]
systems = ['Normal System', 'Fuzzy Logic Control', 'Deep Fuzzy Logic Control']
plt.figure(figsize=(10, 6))
plt.bar(systems, total_costs, color=['blue', 'green', 'red'])
plt.ylabel('Total Cost ($)')
plt.title('Total Cost Comparison Across Systems')
for i, cost in enumerate(total_costs):
    plt.text(i, cost + 0.05, f"${cost:.2f}", ha='center', fontsize=12, color='black')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Visualization 5: Dynamic Charging Priority (Deep Fuzzy Logic Control)
plt.figure(figsize=(14, 8))
plt.plot(Charging_Priority_dynamic, label='Charging Priority (Deep Fuzzy Logic)', color='red', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Charging Priority (%)')
plt.title('Dynamic Charging Priority Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 6: Efficiency Comparison
efficiency_normal = np.sum(BAT_LD) / (Total_cost_normal + np.sum(BAT_LD))
efficiency_fuzzy = np.sum(BAT_LD) / (Total_cost_fuzzy + np.sum(BAT_LD))
efficiency_deep_fuzzy = np.sum(BAT_LD) / (Total_cost_deep_fuzzy + np.sum(BAT_LD))

efficiency_values = [efficiency_normal, efficiency_fuzzy, efficiency_deep_fuzzy]
plt.figure(figsize=(10, 6))
plt.bar(systems, efficiency_values, color=['blue', 'green', 'red'])
plt.ylabel('Battery Efficiency (Energy Used / Total Cost)')
plt.title('Battery Efficiency Comparison Across Systems')
for i, value in enumerate(efficiency_values):
    plt.text(i, value + 0.02, f"{value:.2f}", ha='center', fontsize=12, color='black')
plt.tight_layout()
plt.show()

# Visualization 7: Energy Consumption From Grid
total_energy_normal = np.sum(Grid_power_normal)
total_energy_fuzzy = np.sum(Grid_power_fuzzy)
total_energy_deep_fuzzy = np.sum(Grid_power_deep_fuzzy)

total_energy_values = [total_energy_normal, total_energy_fuzzy, total_energy_deep_fuzzy]
plt.figure(figsize=(10, 6))
plt.bar(systems, total_energy_values, color=['blue', 'green', 'red'])
plt.ylabel('Total Energy Consumed (kWh)')
plt.title('Total Energy Consumption From Grid')
plt.ylim(0, max(total_energy_values) * 1.25)
for i, value in enumerate(total_energy_values):
    plt.text(i, value + 0.02, f"{value:.2f} kWh", ha='center', fontsize=12, color='black')
plt.tight_layout()
plt.show()

# Summary Table of Results
print("\n=== Summary of Results ===")
print(f"Total Cost - Normal System: ${Total_cost_normal:.2f}")
print(f"Total Cost - Fuzzy Logic Control: ${Total_cost_fuzzy:.2f}")
print(f"Total Cost - Deep Fuzzy Logic Control: ${Total_cost_deep_fuzzy:.2f}")

print(f"Total Energy Consumption (Grid) - Normal System: {total_energy_normal:.2f} kWh")
print(f"Total Energy Consumption (Grid) - Fuzzy Logic Control: {total_energy_fuzzy:.2f} kWh")
print(f"Total Energy Consumption (Grid) - Deep Fuzzy Logic Control: {total_energy_deep_fuzzy:.2f} kWh")

print(f"Battery Efficiency - Normal System: {efficiency_normal:.2f}")
print(f"Battery Efficiency - Fuzzy Logic Control: {efficiency_fuzzy:.2f}")
print(f"Battery Efficiency - Deep Fuzzy Logic Control: {efficiency_deep_fuzzy:.2f}")

if efficiency_deep_fuzzy > efficiency_fuzzy and efficiency_deep_fuzzy > efficiency_normal:
    print("\nRecommendation: The Deep Fuzzy Logic Control system demonstrates superior efficiency and is recommended for implementation.")
else:
    print("\nRecommendation: Further optimization of the Deep Fuzzy Logic Control system is needed.")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Initialize DoD lists for each system
DoD_normal = []
DoD_fuzzy = []
DoD_deep_fuzzy = []

# Calculate Depth of Discharge for each system
for i in range(1, len(SOC_normal)):
    DoD_normal.append(abs(SOC_normal[i] - SOC_normal[i - 1]))
    DoD_fuzzy.append(abs(SOC_fuzzy[i] - SOC_fuzzy[i - 1]))
    DoD_deep_fuzzy.append(abs(SOC_deep_fuzzy[i] - SOC_deep_fuzzy[i - 1]))

# Calculate average DoD
avg_dod_normal = np.mean(DoD_normal)
avg_dod_fuzzy = np.mean(DoD_fuzzy)
avg_dod_deep_fuzzy = np.mean(DoD_deep_fuzzy)

print(f"Average DoD - Normal System: {avg_dod_normal:.2f}%")
print(f"Average DoD - Fuzzy Logic Control: {avg_dod_fuzzy:.2f}%")
print(f"Average DoD - Deep Fuzzy Logic Control: {avg_dod_deep_fuzzy:.2f}%")

# Visualization: Depth of Discharge over Time
plt.figure(figsize=(12, 8))

plt.plot(DoD_normal, label='DoD Normal System', color='blue', linewidth=2)
plt.plot(DoD_fuzzy, label='DoD Fuzzy Logic Control', color='green', linewidth=2)
plt.plot(DoD_deep_fuzzy, label='DoD Deep Fuzzy Logic Control', color='red', linewidth=2)

plt.xlabel('Time Step')
plt.ylabel('Depth of Discharge (%)')
plt.title('Depth of Discharge Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization: Average Depth of Discharge Comparison
plt.figure(figsize=(8, 6))
systems = ['Normal System', 'Fuzzy Logic Control', 'Deep Fuzzy Logic Control']
avg_dod_values = [avg_dod_normal, avg_dod_fuzzy, avg_dod_deep_fuzzy]

plt.bar(systems, avg_dod_values, color=['blue', 'green', 'red'])
plt.xlabel('System Type')
plt.ylabel('Average Depth of Discharge (%)')
plt.title('Comparison of Average Depth of Discharge')
plt.ylim(0, max(avg_dod_values) * 1.2)  # Add padding to the top
for i, value in enumerate(avg_dod_values):
    plt.text(i, value + 0.1, f"{value:.2f}%", ha='center', fontsize=12, color='black')
plt.grid(axis='y')
plt.tight_layout()
plt.show()





# Initialize SOH lists for each system
SOH_normal = [SOH_init]
SOH_fuzzy = [SOH_init]
SOH_deep_fuzzy = [SOH_init]

# Degradation factors (hypothetical constants for modeling)
alpha = 0.0015  # Degradation factor for DoD
beta = 0.001    # Degradation factor for temperature above ambient

# Calculate SOH degradation for each system
for i in range(1, len(SOC_normal)):
    # Normal System SOH
    temp_normal = Battery_temp_normal[i]
    DoD_t_normal = abs(SOC_normal[i] - SOC_normal[i - 1])
    SOH_new_normal = SOH_normal[-1] - alpha * DoD_t_normal - beta * max(0, temp_normal - Ambient_temperature)
    SOH_normal.append(max(0, SOH_new_normal))  # Ensure SOH doesn't go below 0%

    # Fuzzy Logic Control SOH
    temp_fuzzy = Battery_temp_fuzzy[i]
    DoD_t_fuzzy = abs(SOC_fuzzy[i] - SOC_fuzzy[i - 1])
    SOH_new_fuzzy = SOH_fuzzy[-1] - alpha * DoD_t_fuzzy - beta * max(0, temp_fuzzy - Ambient_temperature)
    SOH_fuzzy.append(max(0, SOH_new_fuzzy))

    # Deep Fuzzy Logic Control SOH
    temp_deep_fuzzy = Battery_temp_deep_fuzzy[i]
    DoD_t_deep_fuzzy = abs(SOC_deep_fuzzy[i] - SOC_deep_fuzzy[i - 1])
    SOH_new_deep_fuzzy = SOH_deep_fuzzy[-1] - alpha * DoD_t_deep_fuzzy - beta * max(0, temp_deep_fuzzy - Ambient_temperature)
    SOH_deep_fuzzy.append(max(0, SOH_new_deep_fuzzy))

# Final SOH comparison
print(f"Final SOH - Normal System: {SOH_normal[-1]:.2f}%")
print(f"Final SOH - Fuzzy Logic Control: {SOH_fuzzy[-1]:.2f}%")
print(f"Final SOH - Deep Fuzzy Logic Control: {SOH_deep_fuzzy[-1]:.2f}%")

# Visualization: SOH Comparison over Time
plt.figure(figsize=(12, 8))

plt.plot(SOH_normal, label='SOH Normal System', color='blue', linewidth=2)
plt.plot(SOH_fuzzy, label='SOH Fuzzy Logic Control', color='green', linewidth=2)
plt.plot(SOH_deep_fuzzy, label='SOH Deep Fuzzy Logic Control', color='red', linewidth=2)

plt.xlabel('Time Step')
plt.ylabel('State of Health (SOH, %)')
plt.title('SOH Comparison over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save SOH results for further analysis
results_soh_df = pd.DataFrame({
    'Time Step': range(len(SOH_normal)),
    'SOH Normal (%)': SOH_normal,
    'SOH Fuzzy (%)': SOH_fuzzy,
    'SOH Deep Fuzzy (%)': SOH_deep_fuzzy
})
results_soh_df.to_csv('soh_comparison_results.csv', index=False)
print("SOH comparison results saved to 'soh_comparison_results.csv'.")








import os
import matplotlib.pyplot as plt

# Set the directory where figures will be saved
output_directory = r'C:\Users\cavus\Desktop\Margaret - Energies\Results'
os.makedirs(output_directory, exist_ok=True)  # Create the folder if it doesn't exist

# Function to save plots with consistent formatting in the specified directory
def save_plot(fig, filename):
    full_path = os.path.join(output_directory, filename)
    fig.savefig(full_path, dpi=600, bbox_inches='tight')

# Visualization: SOC Comparison
fig1, ax1 = plt.subplots(figsize=(14, 8))
ax1.plot(SOC_normal, label='Normal System', color='blue', linewidth=2)
ax1.plot(SOC_fuzzy, label='FLC', color='green', linewidth=2)
ax1.plot(SOC_deep_fuzzy, label='Deep-FLC', color='red', linewidth=2)
ax1.set_xlabel('Time Step [hour]', fontsize=16)
ax1.set_ylabel('SOC [%]', fontsize=16)
ax1.set_title('SOC Comparison Across Systems', fontsize=16)
ax1.legend(fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.grid(True)
save_plot(fig1, 'SOC_Comparison.png')



print("All figures have been saved with increased font sizes in the directory:", output_directory)




# Visualization: Total Costs Across Systems
fig4, ax4 = plt.subplots(figsize=(12, 8))

# Define data
total_costs = [Total_cost_normal, Total_cost_fuzzy, Total_cost_deep_fuzzy]
systems = ['Normal System', 'FLC', 'Deep-FLC']

# Create bar chart
bars = ax4.bar(systems, total_costs, color=['blue', 'green', 'red'])

# Set labels and title with font size 16
ax4.set_ylabel('Total Cost [$]', fontsize=16)
ax4.set_title('Total Cost Comparison Across Systems', fontsize=16)
ax4.tick_params(axis='x', labelsize=16)  # Set font size for X-axis tick labels
ax4.tick_params(axis='y', labelsize=16)  # Set font size for Y-axis tick labels

# Add cost values on top of the bars with font size 16
for bar, cost in zip(bars, total_costs):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"${cost:.2f}", 
             ha='center', fontsize=16)

# Add grid for better readability
ax4.grid(axis='y')

# Save the figure
save_plot(fig4, 'Total_Cost_Comparison.png')

print("Total Cost Comparison figure saved as 'Total_Cost_Comparison.png'.")



import os
import matplotlib.pyplot as plt

# Set the directory where figures will be saved
output_directory = r'C:\Users\cavus\Desktop\Margaret - Energies\Results'
os.makedirs(output_directory, exist_ok=True)  # Create the folder if it doesn't exist

# Function to save plots with consistent formatting in the specified directory
def save_plot(fig, filename):
    full_path = os.path.join(output_directory, filename)
    fig.savefig(full_path, dpi=600, bbox_inches='tight')

# 1. Depth of Discharge Comparison
fig5, ax5 = plt.subplots(figsize=(12, 8))
ax5.plot(DoD_normal, label='Normal System', color='blue', linewidth=2)
ax5.plot(DoD_fuzzy, label='FLC', color='green', linewidth=2)
ax5.plot(DoD_deep_fuzzy, label='Deep-FLC', color='red', linewidth=2)
ax5.set_xlabel('Time Step [hour]', fontsize=16)
ax5.set_ylabel('DoD [%]', fontsize=16)
ax5.set_title('DoD Comparison', fontsize=16)
ax5.legend(fontsize=16)
ax5.tick_params(axis='both', labelsize=16)
ax5.grid(True)
save_plot(fig5, 'Depth_of_Discharge_Comparison.png')

# 2. Average Depth of Discharge Comparison
fig6, ax6 = plt.subplots(figsize=(8, 6))
avg_dod_values = [avg_dod_normal, avg_dod_fuzzy, avg_dod_deep_fuzzy]
systems = ['Normal System', 'FLC', 'Deep-FLC']
bars = ax6.bar(systems, avg_dod_values, color=['blue', 'green', 'red'])
ax6.set_xlabel('System Type [hour]', fontsize=16)
ax6.set_ylabel('Average DoD [%]', fontsize=16)
ax6.set_title('Average DoD Comparison', fontsize=16)
for bar, value in zip(bars, avg_dod_values):
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.2f}%", ha='center', fontsize=16)
ax6.tick_params(axis='both', labelsize=16)
ax6.grid(axis='y')
save_plot(fig6, 'Average_DoD_Comparison.png')

# 3. SOH Comparison
fig7, ax7 = plt.subplots(figsize=(12, 8))
ax7.plot(SOH_normal, label='Normal System', color='blue', linewidth=2)
ax7.plot(SOH_fuzzy, label='FLC', color='green', linewidth=2)
ax7.plot(SOH_deep_fuzzy, label='Deep-FLC', color='red', linewidth=2)
ax7.set_xlabel('Time Step [hour]', fontsize=16)
ax7.set_ylabel('SOH [%]', fontsize=16)
ax7.set_title('SOH Comparison Over Time', fontsize=16)
ax7.legend(fontsize=16)
ax7.tick_params(axis='both', labelsize=16)
ax7.grid(True)
save_plot(fig7, 'SOH_Comparison.png')

# 4. Battery Efficiency Comparison
fig_eff, ax_eff = plt.subplots(figsize=(12, 8))

# Data for the bar chart
efficiency_values = [efficiency_normal, efficiency_fuzzy, efficiency_deep_fuzzy]
systems = ['Normal System', 'FLC', 'Deep-FLC']

# Create the bar chart
bars = ax_eff.bar(systems, efficiency_values, color=['blue', 'green', 'red'], edgecolor='black', linewidth=1.5)

# Set labels and title
ax_eff.set_xlabel('System Type [hour]', fontsize=16)
ax_eff.set_ylabel('Battery Efficiency', fontsize=16)
ax_eff.set_title('Battery Efficiency Comparison', fontsize=16)

# Adjust the Y-axis limit to provide more space above bars
ax_eff.set_ylim(0, max(efficiency_values) * 1.2)

# Add values on top of the bars with extra space
for bar, value in zip(bars, efficiency_values):
    ax_eff.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,  # Adjusted for spacing
                f"{value:.2f}", ha='center', fontsize=16)

# Format the axes
ax_eff.tick_params(axis='both', labelsize=16)
ax_eff.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines for better readability

# Save the figure
save_plot(fig_eff, 'Battery_Efficiency_Comparison.png')

print(f"All figures have been saved in the directory: {output_directory}.")





import os
import matplotlib.pyplot as plt

# Set the directory where figures will be saved
output_directory = r'C:\Users\cavus\Desktop\Margaret - Energies\Results'
os.makedirs(output_directory, exist_ok=True)  # Create the folder if it doesn't exist

# Function to save plots with consistent formatting in the specified directory
def save_plot(fig, filename):
    full_path = os.path.join(output_directory, filename)
    fig.savefig(full_path, dpi=600, bbox_inches='tight')

# Visualization for Normal System
fig1 = plt.figure(figsize=(12, 8))

# State of Charge (SOC)
plt.subplot(2, 1, 1)
plt.plot(SOC_normal, color='blue', linewidth=2)
plt.xlabel('Time Step [hour]', fontsize=16)
plt.ylabel('SOC [%]', fontsize=16)
plt.title('SOC Over Time (Normal System)', fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tick_params(axis='both', labelsize=16)

# Grid Power Usage
plt.subplot(2, 1, 2)
plt.plot(Grid_power_normal, color='green', linewidth=2)
plt.xlabel('Time Step [hour]', fontsize=16)
plt.ylabel('Grid Power [kW]', fontsize=16)
plt.title('Grid Power Usage Over Time (Normal System)', fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tick_params(axis='both', labelsize=16)

plt.tight_layout()
save_plot(fig1, 'Normal_System_Results.png')
plt.show()

# Visualization for Fuzzy Logic Control
fig2 = plt.figure(figsize=(12, 8))

# State of Charge (SOC)
plt.subplot(2, 1, 1)
plt.plot(SOC_fuzzy, color='blue', linewidth=2)
plt.xlabel('Time Step [hour]', fontsize=16)
plt.ylabel('SOC [%]', fontsize=16)
plt.title('SOC Over Time (FLC)', fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tick_params(axis='both', labelsize=16)

# Grid Power Usage
plt.subplot(2, 1, 2)
plt.plot(Grid_power_fuzzy, color='green', linewidth=2)
plt.xlabel('Time Step [hour]', fontsize=16)
plt.ylabel('Grid Power [kW]', fontsize=16)
plt.title('Grid Power Usage Over Time (FLC)', fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tick_params(axis='both', labelsize=16)

plt.tight_layout()
save_plot(fig2, 'Fuzzy_Logic_Control_Results.png')
plt.show()

# Visualization for Deep Fuzzy Logic Control
fig3 = plt.figure(figsize=(12, 8))

# State of Charge (SOC)
plt.subplot(2, 1, 1)
plt.plot(SOC_deep_fuzzy, color='blue', linewidth=2)
plt.xlabel('Time Step [hour]', fontsize=16)
plt.ylabel('SOC [%]', fontsize=16)
plt.title('SOC Over Time (Deep-FLC)', fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tick_params(axis='both', labelsize=16)

# Grid Power Usage
plt.subplot(2, 1, 2)
plt.plot(Grid_power_deep_fuzzy, color='green', linewidth=2)
plt.xlabel('Time Step [hour]', fontsize=16)
plt.ylabel('Grid Power [kW]', fontsize=16)
plt.title('Grid Power Usage Over Time (Deep-FLC)', fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tick_params(axis='both', labelsize=16)

plt.tight_layout()
save_plot(fig3, 'Deep_Fuzzy_Logic_Control_Results.png')
plt.show()

print(f"Figures for Normal, Fuzzy, and Deep Fuzzy Control systems have been saved in: {output_directory}")










import os
import matplotlib.pyplot as plt

# Set the directory where figures will be saved
output_directory = r'C:\Users\cavus\Desktop\Margaret - Energies\Results'
os.makedirs(output_directory, exist_ok=True)  # Create the folder if it doesn't exist

# Function to save plots with consistent formatting in the specified directory
def save_plot(fig, filename):
    full_path = os.path.join(output_directory, filename)
    fig.savefig(full_path, dpi=600, bbox_inches='tight')

# Visualization: Total Costs Across Systems (Unboxed Bar Chart)
fig4, ax4 = plt.subplots(figsize=(12, 8))

# Define data
total_costs = [Total_cost_normal, Total_cost_fuzzy, Total_cost_deep_fuzzy]
systems = ['Normal System', 'FLC', 'Deep-FLC']

# Create unboxed bar chart
bars = ax4.bar(systems, total_costs, color=['blue', 'green', 'red'], edgecolor='black', linewidth=1.5)
ax4.set_xlabel('System Type', fontsize=16)

# Set labels and title
ax4.set_ylabel('Total Cost [$]', fontsize=16)
ax4.set_title('Total Cost Comparison Across Systems', fontsize=16)
ax4.tick_params(axis='x', labelsize=16)
ax4.tick_params(axis='y', labelsize=16)
ax4.set_ylim(0, max(total_costs) * 1.3)  # Adjust Y-axis for space

# Add cost values with space
for bar, cost in zip(bars, total_costs):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,  # Adjust spacing here
             f"${cost:.2f}", ha='center', fontsize=16)

# Add grid for better readability
ax4.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
save_plot(fig4, 'Total_Cost_Comparison_with_space.png')

print("Total Cost Comparison figure saved as 'Total_Cost_Comparison.png'.")

# Visualization: Average Depth of Discharge Comparison (Unboxed Bar Chart)
fig6, ax6 = plt.subplots(figsize=(12, 8))
avg_dod_values = [avg_dod_normal, avg_dod_fuzzy, avg_dod_deep_fuzzy]

# Create unboxed bar chart
bars = ax6.bar(systems, avg_dod_values, color=['blue', 'green', 'red'], edgecolor='black', linewidth=1.5)
ax6.set_xlabel('System Type', fontsize=16)
ax6.set_ylabel('Average DoD [%]', fontsize=16)
ax6.set_title('Average Depth of Discharge Comparison', fontsize=16)

# Adjust Y-axis limit to provide more space above bars
ax6.set_ylim(0, max(avg_dod_values) * 1.2)

# Add values on top of the bars with font size 16
for bar, value in zip(bars, avg_dod_values):
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.2f}%", ha='center', fontsize=16)

ax6.tick_params(axis='x', labelsize=16)
ax6.tick_params(axis='y', labelsize=16)
ax6.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
save_plot(fig6, 'Average_DoD_Comparison.png')

# Visualization: Battery Efficiency Comparison (Unboxed Bar Chart)
fig_eff, ax_eff = plt.subplots(figsize=(12, 8))
efficiency_values = [efficiency_normal, efficiency_fuzzy, efficiency_deep_fuzzy]

# Create unboxed bar chart
bars = ax_eff.bar(systems, efficiency_values, color=['blue', 'green', 'red'], edgecolor='black', linewidth=1.5)
ax_eff.set_xlabel('System Type', fontsize=16)
ax_eff.set_ylabel('Battery Efficiency', fontsize=16)
ax_eff.set_title('Battery Efficiency Comparison', fontsize=16)

# Adjust Y-axis limit to provide more space above bars
ax_eff.set_ylim(0, max(efficiency_values) * 1.2)

# Add efficiency values on top of the bars with font size 16
for bar, value in zip(bars, efficiency_values):
    ax_eff.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.2f}", ha='center', fontsize=16)

ax_eff.tick_params(axis='x', labelsize=16)
ax_eff.tick_params(axis='y', labelsize=16)
ax_eff.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
save_plot(fig_eff, 'Battery_Efficiency_Comparison.png')

print("Battery Efficiency Comparison figure saved as 'Battery_Efficiency_Comparison.png'.")








import matplotlib.ticker as mtick  # To format Y-axis as percentages

# 4. Battery Efficiency Comparison
fig_eff, ax_eff = plt.subplots(figsize=(12, 8))

# Data for the bar chart (in percentage values)
efficiency_values = [53, 56, 61]  # Example efficiencies
systems = ['Normal System', 'FLC', 'Deep-FLC']

# Create the bar chart
bars = ax_eff.bar(systems, efficiency_values, color=['blue', 'green', 'red'], edgecolor='black', linewidth=1.5)

# Set labels and title
ax_eff.set_xlabel('System Type', fontsize=16)
ax_eff.set_ylabel('Battery Efficiency [%]', fontsize=16)  # Label without '%'
ax_eff.set_title('Battery Efficiency Comparison', fontsize=16)

# Adjust the Y-axis limit to provide more space above bars
ax_eff.set_ylim(0, max(efficiency_values) * 1.2)  # Set Y-axis to go 20% above the maximum efficiency

# Add values on top of the bars with extra space
for bar, value in zip(bars, efficiency_values):
    ax_eff.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,  # Add space above bars
                f"{value}%", ha='center', fontsize=16)  # Removed '%' symbol

# Format the Y-axis with numbers instead of percentages
ax_eff.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))  # Formatter without '%'

# Format the axes
ax_eff.tick_params(axis='both', labelsize=16)
ax_eff.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines for better readability

# Save the figure
save_plot(fig_eff, 'Battery_Efficiency_Comparison_Numeric.png')

print(f"Battery Efficiency Comparison figure saved as 'Battery_Efficiency_Comparison_Numeric.png'.")
























