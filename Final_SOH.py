# -*- coding: utf-8 -*-
"""
SOH Comparison for Normal and Fuzzy Control Systems
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Load Data
file_path = r'C:\Users\cavus\Desktop\Margaret - Energies\Residential_25 - Short_Full.csv'
data = pd.read_csv(file_path)

# Parameters
Battery_Capacity = 4500  # Battery capacity in mAh
Battery_Capacity_kWh = Battery_Capacity * 3.6 / 1000  # Capacity in kWh
SOC_init = 50  # Initial State of Charge (%)
Battery_temp_init = 25  # Initial battery temperature (°C)
Critical_temp = 60  # Critical temperature (°C)
Charging_efficiency = 0.95
Discharging_efficiency = 0.9
Cost_per_kWh = 0.20  # Grid cost in $/kWh
Peak_hour_multiplier = 2  # Cost multiplier during peak hours
Ambient_temperature = 25  # Ambient temperature in °C

# SOH Parameters
SOH_init = 100  # Initial State of Health (%)
alpha = 0.001  # DoD degradation factor
beta = 0.0005  # Temperature degradation factor

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

# Fuzzy Logic System
SoC = ctrl.Antecedent(np.arange(0, 101, 1), 'SoC')
Load = ctrl.Antecedent(np.arange(0, max(P_LD) + 10, 1), 'Load')
Temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'Temperature')
Charging_Priority = ctrl.Consequent(np.arange(0, 101, 1), 'Charging_Priority')
Grid_Priority = ctrl.Consequent(np.arange(0, 101, 1), 'Grid_Priority')

# Membership Functions
SoC['low'] = fuzz.trimf(SoC.universe, [0, 0, 33])
SoC['medium'] = fuzz.trimf(SoC.universe, [30, 50, 70])
SoC['high'] = fuzz.trimf(SoC.universe, [67, 100, 100])

Load['low'] = fuzz.trimf(Load.universe, [0, 0, max(P_LD) / 3])
Load['medium'] = fuzz.trimf(Load.universe, [max(P_LD) / 4, max(P_LD) / 2, 3 * max(P_LD) / 4])
Load['high'] = fuzz.trimf(Load.universe, [2 * max(P_LD) / 3, max(P_LD), max(P_LD)])

Temperature['low'] = fuzz.trimf(Temperature.universe, [0, 0, 35])
Temperature['medium'] = fuzz.trimf(Temperature.universe, [30, 50, 70])
Temperature['high'] = fuzz.trimf(Temperature.universe, [60, 100, 100])

Charging_Priority['low'] = fuzz.trimf(Charging_Priority.universe, [0, 0, 50])
Charging_Priority['medium'] = fuzz.trimf(Charging_Priority.universe, [25, 50, 75])
Charging_Priority['high'] = fuzz.trimf(Charging_Priority.universe, [50, 100, 100])

Grid_Priority['low'] = fuzz.trimf(Grid_Priority.universe, [0, 0, 50])
Grid_Priority['medium'] = fuzz.trimf(Grid_Priority.universe, [25, 50, 75])
Grid_Priority['high'] = fuzz.trimf(Grid_Priority.universe, [50, 100, 100])

# Fuzzy Rules
rule1 = ctrl.Rule(SoC['low'] & Load['high'], (Charging_Priority['high'], Grid_Priority['high']))
rule2 = ctrl.Rule(SoC['medium'] & Load['medium'], (Charging_Priority['medium'], Grid_Priority['medium']))
rule3 = ctrl.Rule(SoC['high'] | Temperature['high'], (Charging_Priority['low'], Grid_Priority['low']))

# Create the Fuzzy Control System
cost_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
cost_sim = ctrl.ControlSystemSimulation(cost_ctrl)

# Initialize Metrics
SOC_normal = [SOC_init]
SOC_fuzzy = [SOC_init]
Battery_temp_normal = [Battery_temp_init]
Battery_temp_fuzzy = [Battery_temp_init]
SOH_normal = [SOH_init]
SOH_fuzzy = [SOH_init]

# Simulation Loop
for k in range(len(P_LD)):
    ### Normal System ###
    SOC_new_normal = SOC_normal[-1] + (PV_BAT[k] - BAT_LD[k] / Discharging_efficiency) / Battery_Capacity_kWh * 100
    SOC_normal.append(max(0, min(100, SOC_new_normal)))

    temp_normal = Battery_temp_normal[-1] + 0.1 * max(0, P_LD[k] - PV_LD[k])  # Example temperature model
    Battery_temp_normal.append(min(max(Ambient_temperature, temp_normal), Critical_temp))

    DoD_t_normal = abs(SOC_normal[-1] - SOC_normal[-2])
    SOH_new_normal = SOH_normal[-1] - alpha * DoD_t_normal - beta * max(0, Battery_temp_normal[-1] - 25)
    SOH_normal.append(max(0, SOH_new_normal))

    ### Fuzzy Logic System ###
    cost_sim.input['SoC'] = SOC_fuzzy[-1]
    cost_sim.input['Load'] = P_LD[k]
    cost_sim.input['Temperature'] = Battery_temp_fuzzy[-1]
    cost_sim.compute()

    charging_priority = cost_sim.output.get('Charging_Priority', 50)

    SOC_new_fuzzy = SOC_fuzzy[-1] + (PV_BAT[k] * charging_priority / 100 - BAT_LD[k] / Discharging_efficiency) / Battery_Capacity_kWh * 100
    SOC_fuzzy.append(max(0, min(100, SOC_new_fuzzy)))

    temp_fuzzy = Battery_temp_fuzzy[-1] + 0.1 * max(0, P_LD[k] - PV_LD[k]) - 0.05 * charging_priority
    Battery_temp_fuzzy.append(min(max(Ambient_temperature, temp_fuzzy), Critical_temp))

    DoD_t_fuzzy = abs(SOC_fuzzy[-1] - SOC_fuzzy[-2])
    SOH_new_fuzzy = SOH_fuzzy[-1] - alpha * DoD_t_fuzzy - beta * max(0, Battery_temp_fuzzy[-1] - 25)
    SOH_fuzzy.append(max(0, SOH_new_fuzzy))

# Visualization of SOH Trends
plt.figure(figsize=(10, 6))
plt.plot(SOH_normal, label='Normal System SOH', color='blue')
plt.plot(SOH_fuzzy, label='Fuzzy Logic System SOH', color='green')
plt.xlabel('Time Step')
plt.ylabel('State of Health (SOH, %)')
plt.title('Battery State of Health (SOH) Comparison')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Print Results
print(f"Final SOH - Normal System: {SOH_normal[-1]:.2f}%")
print(f"Final SOH - Fuzzy Logic System: {SOH_fuzzy[-1]:.2f}%")
