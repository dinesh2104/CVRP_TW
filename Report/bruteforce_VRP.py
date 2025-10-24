import math
import itertools
from typing import List, Tuple, Set

def parse_vrp_file(filename: str):
    """Parse VRP file in TSPLIB format"""
    coordinates = {}
    demands = {}
    capacity = 0
    depot = None
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('CAPACITY'):
            capacity = int(line.split(':')[1].strip())
        
        elif line == 'NODE_COORD_SECTION':
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(('DEMAND_SECTION', 'DEPOT_SECTION', 'EOF')):
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = int(parts[1])
                    y = int(parts[2])
                    coordinates[node_id] = (x, y)
                i += 1
            continue
        
        elif line == 'DEMAND_SECTION':
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(('DEPOT_SECTION', 'EOF')):
                parts = lines[i].strip().split()
                if len(parts) >= 2:
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    demands[node_id] = demand
                i += 1
            continue
        
        elif line == 'DEPOT_SECTION':
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(('EOF', '-1')):
                depot_line = lines[i].strip()
                if depot_line != '-1' and depot_line.isdigit():
                    depot = int(depot_line)
                i += 1
            continue
        
        i += 1
    
    # Find customers (all nodes except depot)
    customers = [node for node in coordinates.keys() if node != depot]
    customers.sort()  # Sort for consistency
    
    return coordinates, demands, capacity, depot, customers

def euclidean_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two coordinates"""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def create_distance_matrix(coordinates: dict) -> dict:
    """Create distance matrix for all node pairs"""
    distances = {}
    for i in coordinates:
        distances[i] = {}
        for j in coordinates:
            distances[i][j] = euclidean_distance(coordinates[i], coordinates[j])
    return distances

def calculate_route_distance(route: List[int], distances: dict, depot: int) -> float:
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

def is_feasible_route(route: List[int], demands: dict, capacity: int) -> bool:
    """Check if a route is feasible (doesn't exceed capacity)"""
    total_demand = sum(demands[customer] for customer in route)
    return total_demand <= capacity

def generate_all_partitions(customers: List[int]) -> List[List[List[int]]]:
    """Generate all possible partitions of customers into routes"""
    if not customers:
        return [[]]
    
    partitions = []
    n = len(customers)
    
    # Generate all possible ways to partition customers
    # Using Stirling numbers approach - generate all set partitions
    for partition_size in range(1, n + 1):
        # Generate all ways to split customers into partition_size groups
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

def solve_cvrp_exponential(coordinates: dict, demands: dict, capacity: int, 
                          depot: int, customers: List[int]) -> Tuple[List[List[int]], float]:
    """
    Solve CVRP using exponential brute force algorithm
    Returns: (best_routes, best_total_distance)
    """
    distances = create_distance_matrix(coordinates)
    
    best_solution = None
    best_distance = float('inf')
    
    print("Solving CVRP using exponential algorithm...")
    print(f"Customers: {customers}")
    print(f"Capacity: {capacity}")
    print(f"Demands: {[demands[c] for c in customers]}")
    print()
    
    # Generate all possible partitions of customers into routes
    all_partitions = generate_all_partitions(customers)
    
    solutions_checked = 0
    
    for partition in all_partitions:
        # Check if all routes in partition are feasible
        feasible = True
        for route in partition:
            if not is_feasible_route(route, demands, capacity):
                feasible = False
                break
        
        if not feasible:
            continue
        
        # For each feasible partition, try all permutations of each route
        route_permutations = []
        for route in partition:
            route_permutations.append(list(itertools.permutations(route)))
        
        # Generate all combinations of route permutations
        for perm_combination in itertools.product(*route_permutations):
            solutions_checked += 1
            
            # Calculate total distance for this solution
            total_distance = 0
            current_routes = []
            
            for route_perm in perm_combination:
                route_list = list(route_perm)
                current_routes.append(route_list)
                total_distance += calculate_route_distance(route_list, distances, depot)
            
            # Update best solution if this is better
            if total_distance < best_distance:
                best_distance = total_distance
                best_solution = current_routes[:]
                print(f"New best solution found! Distance: {best_distance:.2f}")
                for i, route in enumerate(best_solution):
                    route_demand = sum(demands[c] for c in route)
                    route_dist = calculate_route_distance(route, distances, depot)
                    print(f"  Route {i+1}: {depot} -> {' -> '.join(map(str, route))} -> {depot}")
                    print(f"    Demand: {route_demand}/{capacity}, Distance: {route_dist:.2f}")
                print()
    
    print(f"Total solutions checked: {solutions_checked}")
    return best_solution, best_distance

def main():
    """Main function to solve the VRP instance from file"""
    import sys
    
    # Default filename
    filename = "ch2_data2.txt"
    
    # Check if filename provided as command line argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    try:
        coordinates, demands, capacity, depot, customers = parse_vrp_file(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        print("Please make sure the file exists in the current directory.")
        return
    except Exception as e:
        print(f"Error parsing file '{filename}': {e}")
        return
    
    print("=== CVRP Instance ===")
    print(f"File: {filename}")
    print(f"Depot: {depot} at {coordinates[depot]}")
    print("Customers:")
    for customer in customers:
        print(f"  {customer}: {coordinates[customer]}, demand = {demands[customer]}")
    print(f"Vehicle capacity: {capacity}")
    print()
    
    # Solve using exponential algorithm
    best_routes, best_distance = solve_cvrp_exponential(
        coordinates, demands, capacity, depot, customers
    )
    
    if best_routes is None:
        print("No feasible solution found!")
        return
    
    print("=== OPTIMAL SOLUTION ===")
    print(f"Minimum total distance: {best_distance:.2f}")
    print(f"Number of vehicles used: {len(best_routes)}")
    print("\nRoute details:")
    
    distances = create_distance_matrix(coordinates)
    total_demand_served = 0
    for i, route in enumerate(best_routes):
        route_demand = sum(demands[c] for c in route)
        route_distance = calculate_route_distance(route, distances, depot)
        total_demand_served += route_demand
        
        print(f"Vehicle {i+1}: {depot} -> {' -> '.join(map(str, route))} -> {depot}")
        print(f"  Demand: {route_demand}/{capacity}")
        print(f"  Distance: {route_distance:.2f}")
    
    print(f"\nTotal demand served: {total_demand_served}")
    print(f"Total distance: {best_distance:.2f}")

if __name__ == "__main__":
    main()