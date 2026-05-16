# ----------------------------------------------------------------------------
import json
import sys
import os
import pandas as pd
from docplex.util.environment import get_environment
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import math, itertools, heapq, random, time
# Đọc data
file = r"D:\HK252\DA\Data\25\C101.xlsx"
# Distance
df_dist = pd.read_excel(file, sheet_name="distance_matrix", index_col=0)
distance_matrix = df_dist.values.tolist()

#  Demand 
df_demand = pd.read_excel(file, sheet_name="demands")
df_demand.columns = df_demand.columns.str.strip()  # tránh lỗi khoảng trắng
demands = df_demand["Demand"].tolist()

# Vehicle
df_vehicle = pd.read_excel(file, sheet_name="Vehicle")

num_vehicles = int(df_vehicle.loc[
    df_vehicle["Parameter"] == "num_vehicles", "Value"
].values[0])

vehicle_capacity = int(df_vehicle.loc[
    df_vehicle["Parameter"] == "vehicle_capacity", "Value"
].values[0])

#  Auto define 
n = len(demands) - 1
locations = range(n + 1) 
def run_GAP_model(n, locations, distance_matrix,demands,vehicle_capacity,num_vehicles, **kwargs):
# Create model
    mdl = Model('CVRP_CPLEX')

# Decision variables: x[i,j] = 1 if edge (i,j) is used
    x = mdl.binary_var_dict(((i, j) for i in locations for j in locations if i != j), name='x')

# Sub-tour elimination variables: u[i] is the load after visiting customer i
    u = mdl.continuous_var_dict(locations, lb=0, ub=vehicle_capacity, name='u')

# Objective: minimize total distance
    mdl.minimize(mdl.sum(distance_matrix[i][j] * x[i, j] for i in locations for j in locations if i != j))

# Constraints
# Each customer is entered once
    for j in locations[1:]:
        mdl.add_constraint(mdl.sum(x[i, j] for i in locations if i != j) == 1)

# Each customer is left once
    for i in locations[1:]:
        mdl.add_constraint(mdl.sum(x[i, j] for j in locations if i != j) == 1)

# Number of vehicles leaving/entering depot
    mdl.add_constraint(mdl.sum(x[0, j] for j in locations if j != 0) <= num_vehicles)
    mdl.add_constraint(mdl.sum(x[i, 0] for i in locations if i != 0) <= num_vehicles)

# Sub-tour elimination (Miller-Tucker-Zemlin)
    for i in locations[1:]:
        for j in locations[1:]:
            if i != j:
                mdl.add_indicator(x[i, j], u[i] + demands[j] <= u[j])

# Capacity constraints
    for i in locations[1:]:
        mdl.add_constraint(u[i] >= demands[i])
        mdl.add_constraint(u[i] <= vehicle_capacity)

# Depot has 0 load
    mdl.add_constraint(u[0] == 0)

# Solve
    start = time.time()
    mdl.parameters.mip.strategy.file = 3
    mdl.parameters.workmem = 2048
    mdl.parameters.threads = 6
    mdl.parameters.timelimit = 3600

    with open(r"D:\HK252\DA\Data\LOG\MILP\25C101.log", "w", encoding="utf-8") as f_log:

        solution = mdl.solve(log_output=f_log)

        f_log.write("\n\n===== SOLVE DETAILS =====\n")
        f_log.write(str(mdl.solve_details))

        if solution:
            f_log.write("\n\n===== MILP RESULT =====\n")
            f_log.write(f"Optimal cost: {solution.objective_value}\n")

            for (i, j) in x:
                if x[i, j].solution_value > 0.5:
                    f_log.write(f"{i} -> {j}\n")

            f_log.write(f"Time: {round(time.time() - start, 3)} s\n")
        else:
            f_log.write("\nNo solution\n")

    end = time.time()

# Output solution
    if solution:
        print("Total Distance:", solution.objective_value)
        for i, j in x:
            if x[i, j].solution_value > 0.5:
                print(f"Vehicle travels from {i} to {j}")
    else:
        print("No solution found.")
    obj=solution.objective_value
    return obj
if __name__ == '__main__':
    gap_best_obj = run_GAP_model(n, locations, distance_matrix,demands,vehicle_capacity,num_vehicles)  # upper bound
