import math
import itertools
import pandas as pd
from typing import List, Tuple, Dict

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

def euclidean_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two coordinates"""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def create_distance_matrix(coordinates: Dict) -> Dict:
    """Create distance matrix for all node pairs"""
    distances = {}
    for i in coordinates:
        distances[i] = {}
        for j in coordinates:
            distances[i][j] = euclidean_distance(coordinates[i], coordinates[j])
    return distances

def calculate_route_distance(route: List[int], distances: Dict, depot: int) -> float:
    """Calculate total distance for a route starting and ending at depot"""
    if not route:
        return 0
    
    total_distance = 0
    # Distance from depot to first customer
    total_distance += distances[depot][route[0]]
    
    # Distance between consecutive customers
    for i in range(len(route) - 1):
        total_distance += distances[route[i]][route[i + 1]]
    
    # Distance from last customer back to depot
    total_distance += distances[route[-1]][depot]
    
    return total_distance

def is_feasible_route(route: List[int], demands: Dict, capacity: int, 
                     time_windows: Dict, service_times: Dict, 
                     distances: Dict, depot: int) -> bool:
    """Check if a route is feasible (capacity and time window constraints)"""
    # Check capacity constraint
    total_demand = sum(demands[c] for c in route)
    if total_demand > capacity:
        return False
    
    # Check time window constraints
    current_time = time_windows[depot][0]  # Start at depot's ready time
    current_location = depot
    
    for customer in route:
        # Travel to next customer
        travel_time = (distances[current_location][customer])  # Assuming avg speed 50 units/hour
        arrival_time = current_time + travel_time
        
        # Check if we can arrive before the due time
        if arrival_time > time_windows[customer][1]:
            return False
        
        # Wait if we arrive before ready time
        service_start = max(arrival_time, time_windows[customer][0])
        
        # Complete service and move to next
        current_time = service_start + service_times[customer]
        current_location = customer
    
    # Check if we can return to depot in time
    travel_to_depot = distances[current_location][depot]
    return_time = current_time + travel_to_depot
    
    if return_time > time_windows[depot][1]:
        return False
    
    return True

def calculate_route_info(route: List[int], distances: Dict, depot: int,
                        time_windows: Dict, service_times: Dict) -> Tuple[float, float]:
    """Calculate route distance and completion time"""
    if not route:
        return 0, 0
    
    distance = calculate_route_distance(route, distances, depot)
    
    # Calculate completion time
    current_time = time_windows[depot][0]
    current_location = depot
    
    for customer in route:
        travel_time = distances[current_location][customer]
        arrival_time = current_time + travel_time
        service_start = max(arrival_time, time_windows[customer][0])
        current_time = service_start + service_times[customer]
        current_location = customer
    
    completion_time = current_time + distances[current_location][depot]
    
    return distance, completion_time

def generate_all_partitions(customers: List[int]) -> List[List[List[int]]]:
    """Generate all possible partitions of customers into routes"""
    if not customers:
        return [[]]
    
    partitions = []
    n = len(customers)
    
    # Generate all possible ways to partition customers
    for partition_size in range(1, n + 1):
        for partition in generate_partitions_of_size(customers, partition_size):
            partitions.append(partition)
    
    return partitions

def generate_partitions_of_size(items: List[int], k: int) -> List[List[List[int]]]:
    """Generate all partitions of items into exactly k non-empty subsets"""
    if k == 1:
        return [[items]]
    if k == len(items):
        return [[[item] for item in items]]
    if k > len(items) or k < 1:
        return []
    
    partitions = []
    first = items[0]
    rest = items[1:]
    
    # Case 1: first item forms its own subset
    for partition in generate_partitions_of_size(rest, k - 1):
        partitions.append([[first]] + partition)
    
    # Case 2: first item joins one of the existing subsets
    for partition in generate_partitions_of_size(rest, k):
        for i in range(len(partition)):
            new_partition = [subset[:] for subset in partition]
            new_partition[i] = [first] + new_partition[i]
            partitions.append(new_partition)
    
    return partitions

def solve_cvrptw_exponential(coordinates: Dict, demands: Dict, capacity: int, 
                            depot: int, customers: List[int], time_windows: Dict,
                            service_times: Dict, max_vehicles: int) -> Tuple[List[List[int]], float]:
    """
    Solve CVRPTW using exponential brute force algorithm
    Returns: (best_routes, best_total_distance)
    """
    distances = create_distance_matrix(coordinates)
    
    best_solution = None
    best_distance = float('inf')
    
    print("Solving CVRPTW using exponential brute force algorithm...")
    print(f"Customers: {customers}")
    print(f"Capacity: {capacity}")
    print(f"Max vehicles: {max_vehicles}")
    print(f"Demands: {[demands[c] for c in customers]}")
    print()
    
    # Generate all possible partitions of customers into routes
    all_partitions = generate_all_partitions(customers)
    
    solutions_checked = 0
    feasible_solutions = 0
    
    for partition in all_partitions:
        # Skip if more routes than available vehicles
        if len(partition) > max_vehicles:
            continue
        
        # For each feasible partition, try all permutations of each route
        route_permutations = []
        for route in partition:
            route_permutations.append(list(itertools.permutations(route)))
        
        # Generate all combinations of route permutations
        for perm_combination in itertools.product(*route_permutations):
            solutions_checked += 1
            
            # Check if all routes are feasible (capacity and time windows)
            feasible = True
            for route_perm in perm_combination:
                route_list = list(route_perm)
                if not is_feasible_route(route_list, demands, capacity, 
                                       time_windows, service_times, distances, depot):
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            feasible_solutions += 1
            
            # Calculate total distance for this solution
            total_distance = 0
            current_routes = []
            
            for route_perm in perm_combination:
                route_list = list(route_perm)
                current_routes.append(route_list)
                total_distance += calculate_route_distance(route_list, distances, depot)

            #Print all the routes checked
            print(f"Checked solution with routes: {current_routes}, Total Distance: {total_distance:.2f}")
            
            # Update best solution if this is better
            if total_distance < best_distance:
                best_distance = total_distance
                best_solution = current_routes[:]
                print(f"New best solution found! Distance: {best_distance:.2f}")
                for i, route in enumerate(best_solution):
                    route_demand = sum(demands[c] for c in route)
                    route_dist, route_time = calculate_route_info(
                        route, distances, depot, time_windows, service_times
                    )
                    print(f"  Route {i+1}: {depot} -> {' -> '.join(map(str, route))} -> {depot}")
                    print(f"    Demand: {route_demand}/{capacity}, Distance: {route_dist:.2f}, Time: {route_time:.2f}")
                    # Detailed timeline
                    print("    Timeline:")
                    current_time = time_windows[depot][0]
                    current_loc = depot
                    print(f"      Start at depot {depot} at time {current_time:.2f}")
                    for customer in route:
                        travel_time = distances[current_loc][customer]  # Assuming avg speed 50 units/hour
                        arrival = current_time + travel_time
                        start_service = max(arrival, time_windows[customer][0])
                        wait_time = start_service - arrival
                        print(f"      -> Customer {customer}: arrive {arrival:.2f}, wait {wait_time:.2f}, "
                              f"serve [{start_service:.2f}, {start_service + service_times[customer]:.2f}], "
                              f"TW: {time_windows[customer]}")
                        current_time = start_service + service_times[customer]
                        current_loc = customer
                    return_travel = distances[current_loc][depot]
                    print(f"      -> Return to depot: arrive {current_time + return_travel:.2f}")   
                print()
    
    print(f"Total solutions checked: {solutions_checked}")
    print(f"Feasible solutions found: {feasible_solutions}")
    return best_solution, best_distance

def main():
    """Main function to solve the VRPTW instance from file"""
    import sys
    
    # Default filename
    filename = "toy2.txt"
    
    # Check if filename provided as command line argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    try:
        num_vehicles, capacity, depot, df = read_solomon_file(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        print("Please make sure the file exists in the current directory.")
        return
    except Exception as e:
        print(f"Error parsing file '{filename}': {e}")
        return
    
    # Extract data from dataframe
    coordinates = {}
    demands = {}
    time_windows = {}
    service_times = {}
    
    for _, row in df.iterrows():
        cid = int(row['cust_id'])
        coordinates[cid] = (int(row['x']), int(row['y']))
        demands[cid] = int(row['demand'])
        time_windows[cid] = (int(row['ready']), int(row['due']))
        service_times[cid] = int(row['service'])
    
    customers = [c for c in coordinates.keys() if c != depot]
    
    print("=== CVRPTW Instance ===")
    print(f"File: {filename}")
    print(f"Depot: {depot} at {coordinates[depot]}, Time Window: {time_windows[depot]}")
    print("Customers:")
    for customer in customers:
        print(f"  {customer}: {coordinates[customer]}, demand={demands[customer]}, "
              f"TW={time_windows[customer]}, service={service_times[customer]}")
    print(f"Vehicle capacity: {capacity}")
    print(f"Max vehicles: {num_vehicles}")
    print()
    
    # Solve using exponential algorithm
    best_routes, best_distance = solve_cvrptw_exponential(
        coordinates, demands, capacity, depot, customers, 
        time_windows, service_times, num_vehicles
    )
    
    if best_routes is None:
        print("No feasible solution found!")
        return
    
    print("\n=== OPTIMAL SOLUTION ===")
    print(f"Minimum total distance: {best_distance:.2f}")
    print(f"Number of vehicles used: {len(best_routes)}")
    print("\nRoute details:")
    
    distances = create_distance_matrix(coordinates)
    total_demand_served = 0
    
    for i, route in enumerate(best_routes):
        route_demand = sum(demands[c] for c in route)
        route_distance, completion_time = calculate_route_info(
            route, distances, depot, time_windows, service_times
        )
        total_demand_served += route_demand
        
        print(f"\nVehicle {i+1}: {depot} -> {' -> '.join(map(str, route))} -> {depot}")
        print(f"  Demand: {route_demand}/{capacity}")
        print(f"  Distance: {route_distance:.2f}")
        print(f"  Completion time: {completion_time:.2f}")
        
        # Show detailed timeline
        current_time = time_windows[depot][0]
        current_loc = depot
        print(f"  Timeline:")
        print(f"    Start at depot {depot} at time {current_time:.2f}")
        
        for customer in route:
            travel_time = distances[current_loc][customer]
            arrival = current_time + travel_time
            start_service = max(arrival, time_windows[customer][0])
            wait_time = start_service - arrival
            
            print(f"    -> Customer {customer}: arrive {arrival:.2f}, "
                  f"wait {wait_time:.2f}, serve [{start_service:.2f}, "
                  f"{start_service + service_times[customer]:.2f}], "
                  f"TW: {time_windows[customer]}")
            
            current_time = start_service + service_times[customer]
            current_loc = customer
        
        return_travel = distances[current_loc][depot]
        print(f"    -> Return to depot: arrive {current_time + return_travel:.2f}")
    
    print(f"\nTotal demand served: {total_demand_served}")
    print(f"Total distance: {best_distance:.2f}")

if __name__ == "__main__":
    main()