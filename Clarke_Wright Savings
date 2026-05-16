import math
import time
import pandas as pd

# Đọc dữ liệu
file = r"D:\HK252\DA\Data\25\C101.xlsx"
# Distance
df_dist = pd.read_excel(file, sheet_name="distance_matrix", index_col=0)
distance_matrix = df_dist.values.tolist()
#Demand 
df_demand = pd.read_excel(file, sheet_name="demands")
df_demand.columns = df_demand.columns.str.strip()
demands = df_demand["Demand"].tolist()
#  Vehicle 
df_vehicle = pd.read_excel(file, sheet_name="Vehicle")
num_vehicles = int(df_vehicle.loc[
    df_vehicle["Parameter"] == "num_vehicles", "Value"].values[0])
vehicle_capacity = int(df_vehicle.loc[
    df_vehicle["Parameter"] == "vehicle_capacity", "Value"].values[0])
# CLARKE & WRIGHT
def clarke_wright_savings(distance_matrix, demands, vehicle_capacity, num_vehicles):
    n = len(demands) - 1
    # Khởi tạo mỗi khách 1 route riêng
    routes = {i: [0, i, 0] for i in range(1, n + 1)}
    # Tính savings
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings.append((s, i, j))
    # Sắp xếp giảm dần
    savings.sort(reverse=True)
    # Merge route
    for s, i, j in savings:
        route_i = next((r for r in routes.values() if i in r[1:-1]), None)
        route_j = next((r for r in routes.values() if j in r[1:-1]), None)
        if route_i is None or route_j is None or route_i == route_j:
            continue
        # TH1: nối i -> j
        if route_i[-2] == i and route_j[1] == j:
            new_load = sum(demands[k] for k in route_i[1:-1] + route_j[1:-1])
            if new_load <= vehicle_capacity:
                new_route = route_i[:-1] + route_j[1:]
                key_i = next(k for k, v in routes.items() if v == route_i)
                key_j = next(k for k, v in routes.items() if v == route_j)
                routes[key_i] = new_route
                del routes[key_j]
        # TH2: nối j -> i
        elif route_j[-2] == j and route_i[1] == i:
            new_load = sum(demands[k] for k in route_j[1:-1] + route_i[1:-1])
            if new_load <= vehicle_capacity:
                new_route = route_j[:-1] + route_i[1:]
                key_i = next(k for k, v in routes.items() if v == route_i)
                key_j = next(k for k, v in routes.items() if v == route_j)
                routes[key_i] = new_route
                del routes[key_j]
        # dừng nếu đủ xe
        if len(routes) <= num_vehicles:
            break
    final_routes = list(routes.values())[:num_vehicles]
    # Tính cost
    total_cost = 0
    for r in final_routes:
        for k in range(len(r) - 1):
            total_cost += distance_matrix[r[k]][r[k + 1]]
    return total_cost, final_routes
#  RUN
start = time.time()
cost, routes = clarke_wright_savings(distance_matrix,demands,vehicle_capacity,num_vehicles)
end = time.time()
#  OUTPUT
print("\n===== CLARKE & WRIGHT RESULT =====")
print(" Total cost:", cost)
print("Time:", round(end - start, 4), "s")
print(" Number of routes:", len(routes))
print("\n📦 Routes detail:")
for i, r in enumerate(routes):
    load = sum(demands[k] for k in r)
    route_cost = sum(distance_matrix[r[k]][r[k+1]] for k in range(len(r)-1))
    print(f"Route {i+1}: {r} | Load = {load} | Cost = {route_cost}")