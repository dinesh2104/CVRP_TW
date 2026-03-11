import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def verify_routes(df, routes, capacity):
    for i, route in enumerate(routes):
        load = 0
        for cust_id in route:
            demand = df.loc[df["cust_id"] == cust_id, "demand"].values[0]
            load += demand
        if load > capacity:
            print(f"Route {i + 1} exceeds capacity: {load} > {capacity}")
        else:
            print(f"Route {i + 1} is feasible: Load = {load}")

    # Verify time windows
    for i, route in enumerate(routes):
        time = 0
        for j in range(len(route) - 1):
            cust_id = route[j]
            next_cust_id = route[j + 1]
            time += np.sqrt((df.loc[df["cust_id"] == next_cust_id, "x"].values[0] - df.loc[df["cust_id"] == cust_id, "x"].values[0]) ** 2 + 
                            (df.loc[df["cust_id"] == next_cust_id, "y"].values[0] - df.loc[df["cust_id"] == cust_id, "y"].values[0]) ** 2)
            ready = df.loc[df["cust_id"] == cust_id, "ready"].values[0]
            due = df.loc[df["cust_id"] == cust_id, "due"].values[0]
            service = df.loc[df["cust_id"] == cust_id, "service"].values[0]

            if time < ready:
                time = ready
            elif time > due:
                print(f"Route {i + 1} violates time window at customer {cust_id}: Time = {time}, Due = {due}")
                break

            time += service
    print("All routes verified for time windows.")
    
    # Print the total distance
    total_distance = 0
    for route in routes:
        for j in range(len(route) - 1):
            cust_id = route[j]
            next_cust_id = route[j + 1]

            x1 = df.loc[df["cust_id"] == cust_id, "x"].values[0]
            y1 = df.loc[df["cust_id"] == cust_id, "y"].values[0]
            x2 = df.loc[df["cust_id"] == next_cust_id, "x"].values[0]
            y2 = df.loc[df["cust_id"] == next_cust_id, "y"].values[0]

            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_distance += distance
    print(f"Total distance of all routes: {total_distance:.2f}")
        

def read_solomon_file(filename: str):
    with open(filename, "r") as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines if l.strip()]

    vehicle_line_idx = lines.index("VEHICLE") + 2
    num_vehicles, capacity = map(int, lines[vehicle_line_idx].split())

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

    df = pd.DataFrame(
        cust_data,
        columns=["cust_id", "x", "y", "demand", "ready", "due", "service"]
    )
    return num_vehicles, capacity, 0, df


def plot_clusters(df, clusters):
    """
    Plots the customers colored by their assigned cluster.
    """
    plt.figure(figsize=(12, 10))

    # Distinct vibrant colors for clusters
    color_list = [
        "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
        "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1"
    ]
    colors = color_list[:len(clusters)]

    # Plot depot (square red)
    depot = df.loc[df["cust_id"] == 0]
    depot_x, depot_y = depot["x"].values[0], depot["y"].values[0]
    plt.scatter(depot_x, depot_y, c='red', s=250, marker='s',
                edgecolor='black', label='Depot', zorder=5)
    plt.text(depot_x, depot_y, '0', fontsize=12, weight='bold', color='white', ha='center', va='center', zorder=6)

    # Plot customers by cluster
    for i, cluster_nodes in enumerate(clusters):
        # Filter dataframe for nodes in the current cluster
        cluster_df = df[df["cust_id"].isin(cluster_nodes)]
        
        plt.scatter(cluster_df["x"], cluster_df["y"],
                    c=colors[i], s=250, edgecolor='black', linewidth=1.2,
                    label=f'Cluster {i}', zorder=4)

        # Label the points with their Customer ID
        for _, row in cluster_df.iterrows():
            plt.text(
                row["x"], row["y"],
                str(int(row["cust_id"])),
                fontsize=10, weight='bold',
                color='white', ha='center', va='center',
                zorder=6
            )
            
    # Optional: Plot unassigned customers in grey (if any were missed in the clusters)
    assigned_nodes = [node for cluster in clusters for node in cluster] + [0]
    unassigned_df = df[~df["cust_id"].isin(assigned_nodes)]
    if not unassigned_df.empty:
        plt.scatter(unassigned_df["x"], unassigned_df["y"],
                    c='lightgrey', s=250, edgecolor='black', linewidth=1.2,
                    label='Unassigned', zorder=4)
        for _, row in unassigned_df.iterrows():
            plt.text(row["x"], row["y"], str(int(row["cust_id"])),
                     fontsize=10, weight='bold', color='black', ha='center', va='center', zorder=6)

    plt.title("CVRPTW Customer Clustering", fontsize=16, weight='bold', pad=20)
    plt.xlabel("X coordinate", fontsize=13)
    plt.ylabel("Y coordinate", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.show()


def solve_cvrptw_from_file(filename: str):
    print(f"=== Solving {filename} ===")

    num_vehicles, capacity, depot, df = read_solomon_file(filename)

    # Farest clustering
    # clusters = [
    #     [1, 2, 6, 12, 18, 19, 21, 22, 33, 41, 44, 47, 48],
    #     [4, 8, 9, 10, 24, 27, 28, 37, 38, 50],
    #     [3, 13, 14, 15, 16, 25, 26, 35, 36, 42, 43, 45, 46],
    #     [5, 7, 11, 17, 20, 23, 29, 30, 31, 32, 34, 39, 40, 49]
    # ]

    # # Plot the clusters
    # plot_clusters(df, clusters)

    # # K mean++
    # clusters=[[7, 23, 30, 31, 32, 39, 40],
    #           [4, 5, 8, 9, 10, 11, 24, 26, 27, 28, 29, 34, 37, 38, 49, 50],
    #           [2, 3, 13, 14, 15, 16, 18, 25, 35, 36, 42, 43, 45, 46, 47, 48],
    #           [1, 6, 12, 17, 19, 20, 21, 22, 33, 41, 44]]
    
    # plot_clusters(df, clusters)

    # # Hierarchical clustering
    # clusters=[[1, 20, 17, 34, 7, 39, 40, 12, 19, 5, 29, 11, 49, 23, 32, 30, 31],
    #           [2, 48, 21, 33, 47, 15, 18, 43, 6, 44, 22, 41],
    #           [3, 35, 36, 16, 13, 45, 46, 26, 14, 25, 42],
    #           [4, 8, 38, 50, 27, 37, 9, 28, 10, 24]]
    # plot_clusters(df, clusters)

    clusters=[[1, 20, 17,34],
              [2, 48, 21, 33, 47, 15, 18, 43],
              [3, 35, 36, 16],
              [4, 8, 38, 50, 27, 37],
              [5, 29, 11, 49],
              [6, 44, 22, 41],
              [7, 39, 40, 12, 19],
              [9, 28, 10, 24],
              [13, 45, 46, 26],
              [14, 25, 42],
              [23, 32, 30,31]] 
    plot_clusters(df, clusters)


# Example usage
filename = "r_50.txt" # Change to "c_50.txt" or whatever file has these 50 nodes
solve_cvrptw_from_file(filename)