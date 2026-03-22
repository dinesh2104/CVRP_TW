import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import cudf
import numpy as np
from cuopt import routing
import matplotlib.font_manager as fm
import time

def read_vrp_file(filepath):
    """
    Reads a CVRP file and extracts coordinates after NODE_COORD_SECTION.
    """
    coords = []
    capacity = 0
    name = ""
    with open(filepath, "r") as f:
        lines = f.readlines()

    start_idx = None
    depot = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("NODE_COORD_SECTION"):
            start_idx = i + 1

        if line.strip().startswith("CAPACITY"):
            capacity = int(line.split(":")[1].strip())
            #print(f"Vehicle Capacity: {capacity}")
        if line.strip().startswith("NAME"):
            name = line.split(":")[1].strip()
            #print(f"Problem Name: {name}")
        if line.strip().startswith("DEPOT_SECTION"):
            depot = int(lines[i+1].strip())

    demand_idx=None
    # Read until EOF or 'DEPOT_SECTION' if present
    for line in lines[start_idx:]:
        if line.strip() == "" or line.strip().startswith("DEMAND_SECTION"):
            demand_idx = lines.index(line)
            break
        parts = line.split()
        if len(parts) >= 3:
            node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            coords.append([x, y])

    demands = []

    for line in lines[demand_idx:]:
        if line.strip() == "" or line.strip().startswith("DEPOT_SECTION"):
            break
        parts = line.split()
        if len(parts) >= 2:
            node_id, demand = int(parts[0]), int(parts[1])
            demands.append(demand)

    return name, capacity, coords, demands, depot

folder = "/content/drive/MyDrive/MTP/CVRP_TW/CVRP"
for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(folder, filename)

        name, capacity, location_coordinates, demand_values, depot = read_vrp_file(filepath)
        location_names = [str(i) for i in range(0, len(location_coordinates) )]

        location_coordinates_df = pd.DataFrame(location_coordinates,
                                               columns=['xcord', 'ycord'],
                                               index=location_names)

        
        distance_matrix = distance.cdist(location_coordinates_df.values, location_coordinates_df.values, "euclidean")
        distance_matrix_df = cudf.DataFrame(np.array(distance_matrix).astype(np.float32))
        
        total_demand = sum(demand_values)
        max_vehicles_needed = min(len(demand_values), int(np.ceil(total_demand / capacity)) + 2)

        vehicle_capacity_val = [capacity for i in range(max_vehicles_needed)]
        vehicle_capacity_val = [total_demand*2]
        
        location_demand = cudf.Series(demand_values, dtype=np.int32)
        vehicle_capacity = cudf.Series(vehicle_capacity_val, dtype=np.int32)
        
        n_vehicles = len(vehicle_capacity)
        n_locations = len(location_demand)

        
        data_model = routing.DataModel(n_locations, n_vehicles)
        data_model.add_cost_matrix(distance_matrix_df)
        data_model.add_capacity_dimension("demand", location_demand, vehicle_capacity)

        
        veh_start_locations = cudf.Series([depot for i in range(n_vehicles)])
        veh_end_locations = cudf.Series([depot for i in range(n_vehicles)])
        

        data_model.set_vehicle_locations(veh_start_locations, veh_end_locations)

        solver_settings = routing.SolverSettings()
        solver_settings.set_time_limit(10)

        start_time = time.time()
        solution = routing.Solve(data_model, solver_settings)
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Route calculation completed in {execution_time:.4f} seconds")

        if solution.get_status() == 0:
            print("Cost for the routing in distance: ", solution.get_total_objective())
            print("Vehicle count to complete routing: ", solution.get_vehicle_count())
            print(f"Vehicles actually used: {solution.get_vehicle_count()} out of {n_vehicles} available")

        
            routes_df = solution.get_route()

            for vehicle_id in routes_df['truck_id'].unique().to_pandas():
                vehicle_route = routes_df[routes_df['truck_id'] == vehicle_id]
                route_locations = vehicle_route['route'].to_arrow().to_pylist()
                route_names = [location_names[loc] for loc in route_locations]

                route_demand = sum(demand_values[loc] for loc in route_locations if loc != depot)
                print(f"Vehicle {vehicle_id} route: {' → '.join(route_names)} (Demand: {route_demand}/{capacity})")
        else:
            print("NVIDIA cuOpt Failed to find a feasible solution. Status:", solution.get_status())
            print("Error Message:",solution.get_error_message())
