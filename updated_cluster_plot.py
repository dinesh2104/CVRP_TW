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
    Plots the customers colored by their assigned cluster, 
    with the Depot strictly centered at (0,0).
    """
    # Create a copy so we don't permanently alter the original coordinates
    plot_df = df.copy()

    # Find original depot coordinates
    orig_depot_x = plot_df.loc[plot_df["cust_id"] == 0, "x"].values[0]
    orig_depot_y = plot_df.loc[plot_df["cust_id"] == 0, "y"].values[0]

    # Translate all points so the depot becomes (0, 0)
    plot_df["x"] = plot_df["x"] - orig_depot_x
    plot_df["y"] = plot_df["y"] - orig_depot_y

    plt.figure(figsize=(12, 10))

    # Distinct vibrant colors for clusters
    color_list = [
        "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
        "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1"
    ]
    colors = color_list[:len(clusters)]

    # Draw crosshairs (X and Y axis) passing through the new depot center (0,0)
    plt.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=1)
    plt.axvline(0, color='black', linewidth=1.5, linestyle='-', zorder=1)

    # Plot depot (square red, now at 0,0)
    plt.scatter(0, 0, c='red', s=300, marker='s', edgecolor='black', label='Depot (Centered)', zorder=5)
    plt.text(0, 0, '0', fontsize=12, weight='bold', color='white', ha='center', va='center', zorder=6)

    # Plot customers by cluster
    for i, cluster_nodes in enumerate(clusters):
        # Filter dataframe for nodes in the current cluster
        cluster_df = plot_df[plot_df["cust_id"].isin(cluster_nodes)]
        
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
            
    # Optional: Plot unassigned customers in grey
    assigned_nodes = [node for cluster in clusters for node in cluster] + [0]
    unassigned_df = plot_df[~plot_df["cust_id"].isin(assigned_nodes)]
    if not unassigned_df.empty:
        plt.scatter(unassigned_df["x"], unassigned_df["y"],
                    c='lightgrey', s=250, edgecolor='black', linewidth=1.2,
                    label='Unassigned', zorder=4)
        for _, row in unassigned_df.iterrows():
            plt.text(row["x"], row["y"], str(int(row["cust_id"])),
                     fontsize=10, weight='bold', color='black', ha='center', va='center', zorder=6)

    plt.title("CVRPTW Customer Clustering (Depot-Centric View)", fontsize=16, weight='bold', pad=20)
    plt.xlabel(f"Relative X coordinate (Original offset: {orig_depot_x})", fontsize=13)
    plt.ylabel(f"Relative Y coordinate (Original offset: {orig_depot_y})", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # Make sure axes are symmetrical so circles don't look like ovals
    plt.axis('equal') 
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.show()


def solve_cvrptw_from_file(filename: str):
    print(f"=== Solving {filename} ===")

    num_vehicles, capacity, depot, df = read_solomon_file(filename)

    # ---------------Plot for sweep----------

    # Total cost = 987.334 angle = 45
    clusters=[[4, 10, 38, 28, 24, 50, 9, 8, 27],
              [37, 26, 46, 45, 14, 13],
              [42, 25, 3, 16, 36, 35, 15],
              [43, 18, 2, 47, 48, 22, 21, 33, 6],
              [44, 41, 1, 12, 19, 17],
              [40, 20, 39, 7, 32, 34, 30, 31, 23],
              [11, 5, 29, 49]]

    plot_clusters(df, clusters)

    # Total cost = 1002.28 & angle = 45
    clusters=[[46, 45, 14, 13, 42, 25, 3, 16, 36],
              [35, 15, 43, 18, 2, 47],
              [48, 22, 21, 33, 6, 44, 41, 1, 12],
              [19, 17, 40, 20, 39, 7],
              [32, 34, 30, 31, 23, 11, 5, 29, 49],
              [4, 10, 38, 28, 24, 50, 9, 8, 27],
              [37, 26 ]]
    plot_clusters(df,clusters)

    # Based on demand
    clusters=[[41, 1, 12, 19, 17, 40, 20, 39, 7, 32, 34],
              [30, 31, 23, 11, 5, 29, 49, 4, 10, 38, 28, 24],
              [50, 9, 8, 27, 37, 26, 46, 45, 14, 13, 42],
              [25, 3, 16, 36, 35, 15, 43, 18, 2, 47, 48, 22],
              [21, 33, 6, 44]]
    plot_clusters(df,clusters)

    clusters=[[17, 40, 20, 39, 7, 32, 34, 30, 31, 23],
              [11, 5, 29, 49, 4, 10, 38, 28, 24, 50, 9, 8, 27],
              [37, 26, 46, 45, 14, 13, 42, 25, 3, 16, 36, 35, 15],
              [43, 18, 2, 47, 48, 22, 21, 33, 6, 44, 41, 1],
              [12, 19]]
    plot_clusters(df,clusters)

    clusters=[[39, 7, 32, 34, 30, 31, 23, 11, 5, 29, 49, 4],
              [10, 38, 28, 24, 50, 9, 8, 27, 37, 26, 46, 45],
              [14, 13, 42, 25, 3, 16, 36, 35, 15, 43, 18],
              [2, 47, 48, 22, 21, 33, 6, 44, 41, 1, 12, 19, 17]]
    plot_clusters(df,clusters)

    clusters=[[39, 7, 32, 34, 30, 31, 23, 11, 5, 29, 49, 4],
              [10, 38, 28, 24, 50, 9, 8, 27, 37, 26, 46, 45],
              [14, 13, 42, 25, 3, 16, 36, 35, 15, 43, 18],
              [2, 47, 48, 22, 21, 33, 6, 44, 41, 1, 12, 19, 17],
              [40, 20]]
    plot_clusters(df,clusters)

    clusters=[[39, 7, 32, 34, 30, 31, 23, 11, 5, 29, 49, 4],
              [14, 13, 42, 25, 3, 16, 36, 35, 15, 43, 18],
              [2, 47, 48, 22, 21, 33, 6, 44, 41, 1, 12, 19, 17],
              [40, 20, 39]]
    plot_clusters(df,clusters)

# Example usage
filename = "r_50.txt" # Ensure this file is in your directory
solve_cvrptw_from_file(filename)