import pandas as pd
import numpy as np
from scipy.spatial import distance
import cudf
from cuopt import routing
import time

def read_solomon_file(filename: str):
    """
    Parse Solomon VRPTW instance file (like C101.txt).
    Returns depot index, vehicle count, vehicle capacity,
    location coordinates, demands, time windows, and service times.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Strip and split
    lines = [l.strip() for l in lines if l.strip()]

    # Vehicle info (after 'VEHICLE' line)
    vehicle_line_idx = lines.index("VEHICLE") + 2
    num_vehicles, capacity = map(int, lines[vehicle_line_idx].split())

    # Customer info (after 'CUSTOMER' line + header)
    cust_idx = lines.index("CUSTOMER") + 2
    customer_lines = lines[cust_idx:]

    cust_data = []
    for line in customer_lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
        x, y = int(parts[1]), int(parts[2])
        demand = int(parts[3])
        ready = int(parts[4])
        due = int(parts[5])
        service = int(parts[6])
        cust_data.append((cid, x, y, demand, ready, due, service))

    df = pd.DataFrame(cust_data,
                      columns=["cust_id", "x", "y", "demand", "ready", "due", "service"])

    depot = 0  # always first entry in Solomon format

    return num_vehicles, capacity, depot, df
  
def calculate_route_distance(route_locations, distance_matrix_df):
    total_distance = 0.0
    for i in range(len(route_locations) - 1):
        from_location = route_locations[i]
        to_location = route_locations[i + 1]
        # Access distance directly (convert to Python float)
        total_distance += float(distance_matrix_df.iloc[from_location, to_location])
    return total_distance

def print_detailed_route_info(routes_df, demand_values, distance_matrix_df, depot, capacity):
    total_system_distance = 0.0
    total_system_demand = 0

    print("\n" + "="*80)
    print("DETAILED ROUTE ANALYSIS")
    print("="*80)

    for vehicle_id in routes_df['truck_id'].unique().to_pandas():
        vehicle_route = routes_df[routes_df['truck_id'] == vehicle_id]
        route_locations = vehicle_route['route'].to_arrow().to_pylist()

        # Total demand
        route_demand = sum(demand_values[loc] for loc in route_locations if loc != depot)

        # Total distance
        route_distance = calculate_route_distance(route_locations, distance_matrix_df)

        total_system_distance += route_distance
        total_system_demand += route_demand

        print(f"\nVehicle {vehicle_id}:")
        print(f"  Route: {' → '.join(str(loc) for loc in route_locations)}")
        #print(f"  Total demand served: {route_demand}/{capacity}")
        #print(f"  Capacity utilization: {(route_demand/capacity)*100:.1f}%")
        print(f"  Total distance traveled: {route_distance:.2f}")

        # print(f"  Step-by-step distances:")
        # for i in range(len(route_locations) - 1):
        #     from_loc = route_locations[i]
        #     to_loc = route_locations[i + 1]
        #     step_distance = float(distance_matrix_df.iloc[from_loc, to_loc])
        #     print(f"    {from_loc} → {to_loc}: {step_distance:.2f}")

    print(f"\nSYSTEM SUMMARY:")
    print(f"Total vehicles used: {len(routes_df['truck_id'].unique())}")
    print(f"Total system distance: {total_system_distance:.2f}")
    print(f"Total demand served: {total_system_demand}")
    print(f"Average distance per vehicle: {total_system_distance/len(routes_df['truck_id'].unique()):.2f}")

    return total_system_distance

def solve_cvrptw_from_file(filename: str):
    """
    Solve CVRPTW from Solomon benchmark file using cuOpt.
    """
    print(f"=== Solving {filename} ===")

    # Read data
    num_vehicles, capacity, depot, df = read_solomon_file(filename)

    # Locations
    location_coordinates = df[["x", "y"]].values.tolist()
    demand_values = df["demand"].tolist()
    time_windows = df[["ready", "due"]].values.tolist()
    service_times = df["service"].tolist()

    # Distance matrix
    distance_matrix = distance.cdist(location_coordinates, location_coordinates, "euclidean")
    distance_matrix_df = cudf.DataFrame(np.array(distance_matrix).astype(np.float32))

    # Vehicle capacities
    vehicle_capacity = cudf.Series([capacity] * num_vehicles, dtype=np.int32)

    # Convert to cuDF Series
    location_demand = cudf.Series(demand_values, dtype=np.int32)
    earliest_times = cudf.Series(df["ready"].tolist(), dtype=np.int32)
    latest_times = cudf.Series(df["due"].tolist(), dtype=np.int32)
    service_times_series = cudf.Series(service_times, dtype=np.int32)

    # Data model
    n_locations = len(df)
    data_model = routing.DataModel(n_locations, num_vehicles)
    data_model.add_cost_matrix(distance_matrix_df)
    data_model.add_capacity_dimension("demand", location_demand, vehicle_capacity)

    # Add time windows + service times
    data_model.set_order_time_windows(earliest_times, latest_times)
    data_model.set_order_service_times(service_times_series)

    # Start/end locations (all vehicles at depot)
    starts = cudf.Series([depot] * num_vehicles, dtype=np.int32)
    ends = cudf.Series([depot] * num_vehicles, dtype=np.int32)
    data_model.set_vehicle_locations(starts, ends)

    # Solver settings
    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(60)  # 60s for benchmark instance

    # Solve
    print("\n=== Solving CVRPTW ===")
    start = time.time()
    solution = routing.Solve(data_model, solver_settings)
    end = time.time()
    print(f"Execution time: {end - start:.2f}s")

    if solution.get_status() == 0:
        print(f"\n=== SOLUTION FOUND ===")
        print(f"Total cost: {solution.get_total_objective():.2f}")
        print(f"Vehicles used: {solution.get_vehicle_count()}")
        routes_df = solution.get_route()

        for vid in sorted(routes_df['truck_id'].unique().to_pandas()):
            vehicle_route = routes_df[routes_df['truck_id'] == vid]
            route = vehicle_route['route'].to_arrow().to_pylist()
            if len(route) <= 2:
                continue
            print(f"\nVehicle {vid}: Route {route}")

        actual_total_distance = print_detailed_route_info(
                routes_df, demand_values, distance_matrix_df, depot, capacity
            )

    else:
        print("\n=== NO SOLUTION ===")
        print(f"Status: {solution.get_status()}")
        print(f"Message: {solution.get_error_message()}")

if __name__ == "__main__":
    solve_cvrptw_from_file("/content/drive/MyDrive/MTP/CVRP_TW/inputs/c101.txt")
