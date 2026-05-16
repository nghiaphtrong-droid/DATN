import pandas as pd
import time
from docplex.mp.model import Model

#  Đọc DATA
file = r"D:\HK252\DA\Data\25\C101.xlsx"

df_dist = pd.read_excel(file, sheet_name="distance_matrix", index_col=0)
distance_matrix = df_dist.values.tolist()

df_demand = pd.read_excel(file, sheet_name="demands")
df_demand.columns = df_demand.columns.str.strip()
demands = df_demand["Demand"].tolist()

df_vehicle = pd.read_excel(file, sheet_name="Vehicle")

num_vehicles = int(df_vehicle.loc[
    df_vehicle["Parameter"] == "num_vehicles", "Value"
].values[0])

vehicle_capacity = int(df_vehicle.loc[
    df_vehicle["Parameter"] == "vehicle_capacity", "Value"
].values[0])

n = len(demands) - 1
locations = range(n + 1)

# 🚚 CLARKE-WRIGHT
def clarke_wright_savings(distance_matrix, demands, vehicle_capacity, num_vehicles):

    n = len(demands) - 1
    routes = {i: [0, i, 0] for i in range(1, n + 1)}

    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings.append((s, i, j))

    savings.sort(reverse=True)

    for s, i, j in savings:

        route_i = next((r for r in routes.values() if i in r[1:-1]), None)
        route_j = next((r for r in routes.values() if j in r[1:-1]), None)

        if route_i is None or route_j is None or route_i == route_j:
            continue

        if route_i[-2] == i and route_j[1] == j:
            new_load = sum(demands[k] for k in route_i[1:-1] + route_j[1:-1])

            if new_load <= vehicle_capacity:
                new_route = route_i[:-1] + route_j[1:]
                key_i = next(k for k, v in routes.items() if v == route_i)
                key_j = next(k for k, v in routes.items() if v == route_j)
                routes[key_i] = new_route
                del routes[key_j]

        elif route_j[-2] == j and route_i[1] == i:
            new_load = sum(demands[k] for k in route_j[1:-1] + route_i[1:-1])

            if new_load <= vehicle_capacity:
                new_route = route_j[:-1] + route_i[1:]
                key_i = next(k for k, v in routes.items() if v == route_i)
                key_j = next(k for k, v in routes.items() if v == route_j)
                routes[key_i] = new_route
                del routes[key_j]

        if len(routes) <= num_vehicles:
            break

    final_routes = list(routes.values())[:num_vehicles]

    total_cost = 0
    for r in final_routes:
        for k in range(len(r) - 1):
            total_cost += distance_matrix[r[k]][r[k + 1]]

    return total_cost, final_routes

#  RUN CW
cw_cost, routes = clarke_wright_savings(
    distance_matrix, demands, vehicle_capacity, num_vehicles
)

print("\n===== CLARKE & WRIGHT =====")
print("Cost:", cw_cost)
print("Routes:", routes)

#  MILP MODEL
def run_model(n, locations, distance_matrix,demands,vehicle_capacity,num_vehicles,**kwargs):

    mdl = Model('CVRP')

    x = mdl.binary_var_dict(
        ((i, j) for i in locations for j in locations if i != j), name='x'
    )

    u = mdl.continuous_var_dict(locations, lb=0, ub=vehicle_capacity, name='u')

    f = mdl.continuous_var_dict(
        ((i, j) for i in locations for j in locations if i != j),
        lb=0, ub=vehicle_capacity, name='f'
    )

    # Objective
    mdl.minimize(
        mdl.sum(distance_matrix[i][j] * x[i, j]
        for i in locations for j in locations if i != j)
    )

    # Constraints
    for j in locations[1:]:
        mdl.add_constraint(mdl.sum(x[i, j] for i in locations if i != j) == 1)

    for i in locations[1:]:
        mdl.add_constraint(mdl.sum(x[i, j] for j in locations if i != j) == 1)

    mdl.add_constraint(mdl.sum(x[0, j] for j in locations if j != 0) <= num_vehicles)
    mdl.add_constraint(mdl.sum(x[i, 0] for i in locations if i != 0) <= num_vehicles)

    for i in locations:
        for j in locations:
            if i != j:
                mdl.add_constraint(f[i, j] <= vehicle_capacity * x[i, j])

    for j in locations[1:]:
        mdl.add_constraint(
            mdl.sum(f[i, j] for i in locations if i != j) -
            mdl.sum(f[j, k] for k in locations if k != j)
            == demands[j])
    mdl.add_constraint(
        mdl.sum(f[0, j] for j in locations if j != 0) == sum(demands)
    )
    for i in locations[1:]:
        for j in locations[1:]:
            if i != j:
                mdl.add_indicator(x[i, j], u[i] + demands[j] <= u[j])

    for i in locations[1:]:
        mdl.add_constraint(u[i] >= demands[i])
        mdl.add_constraint(u[i] <= vehicle_capacity)

    mdl.add_constraint(u[0] == 0)
#  MIP START FROM CW
    mip_start = mdl.new_solution()

    # x = 0
    for (i, j) in x:
        mip_start.add_var_value(x[i, j], 0)

    # set route
    for route in routes:
        for k in range(len(route) - 1):
            i = route[k]
            j = route[k + 1]
            if (i, j) in x:
                mip_start.add_var_value(x[i, j], 1)

    # u
    for route in routes:
        load = 0
        for node in route:
            if node != 0:
                load += demands[node]
                mip_start.add_var_value(u[node], load)

    mip_start.add_var_value(u[0], 0)

    for (i, j) in f:
        mip_start.add_var_value(f[i, j], 0)

    for route in routes:

        customers = route[1:-1]

        remaining = sum(demands[c] for c in customers)

    for k in range(len(route) - 1):

        i = route[k]
        j = route[k + 1]

        if (i, j) in f:
            mip_start.add_var_value(f[i, j], remaining)

        if j != 0:
            remaining -= demands[j]

    mdl.add_mip_start(mip_start)

    # SOLVE + SAVE LOG
    start = time.time()
    mdl.parameters.mip.strategy.file = 3
    mdl.parameters.workmem = 2048
    mdl.parameters.threads = 6
    mdl.parameters.timelimit = 7200
    mdl.parameters.mip.tolerances.uppercutoff = 259
    mdl.parameters.mip.tolerances.lowercutoff = 193

    with open(r"D:\HK252\DA\Data\LOG\CUT\200R121.log", "w", encoding="utf-8") as f_log:

        sol = mdl.solve(log_output=f_log)

        f_log.write("\n\n===== SOLVE DETAILS =====\n")
        f_log.write(str(mdl.solve_details))

        if sol:
            f_log.write("\n\n===== MILP RESULT =====\n")
            f_log.write(f"Optimal cost: {sol.objective_value}\n")

            for (i, j) in x:
                if x[i, j].solution_value > 0.5:
                    f_log.write(f"{i} -> {j}\n")

            f_log.write(f"Time: {round(time.time() - start, 3)} s\n")
        else:
            f_log.write("\nNo solution\n")

    end = time.time()

    # Print ra màn hình
    if sol:
        print("\n===== MILP RESULT =====")
        print("Optimal cost:", sol.objective_value)

        for (i, j) in x:
            if x[i, j].solution_value > 0.5:
                print(f"{i} -> {j}")

        print("Time:", round(end - start, 3), "s")
    else:
        print("No solution")

# ▶️ RUN
run_model(n,locations,distance_matrix,demands,vehicle_capacity,num_vehicles)