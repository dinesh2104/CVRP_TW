import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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

    #Verify time windows
    for i, route in enumerate(routes):
        time = 0
        for j in range(len(route) - 1):
            cust_id = route[j]
            next_cust_id = route[j + 1]
            time+= np.sqrt((df.loc[df["cust_id"] == next_cust_id, "x"].values[0] - df.loc[df["cust_id"] == cust_id, "x"].values[0]) ** 2 + (df.loc[df["cust_id"] == next_cust_id, "y"].values[0] - df.loc[df["cust_id"] == cust_id, "y"].values[0]) ** 2)
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


def plot_routes(df, routes):
    plt.figure(figsize=(12, 10))

    # Distinct vibrant colors for routes
    color_list = [
        "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
        "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
        "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff"
    ]
    colors = color_list[:len(routes)]

    # Plot depot (square red)
    depot = df.loc[df["cust_id"] == 0]
    depot_x, depot_y = depot["x"].values[0], depot["y"].values[0]
    plt.scatter(depot_x, depot_y, c='red', s=250, marker='s',
                edgecolor='black', label='Depot', zorder=5)

    # Plot customers (round grey)
    cust_df = df[df["cust_id"] != 0]
    plt.scatter(cust_df["x"], cust_df["y"],
                c='lightgrey', s=180, edgecolor='black', linewidth=1.2,
                label='Customers', zorder=4)

    # Label all points (black number inside grey circle)
    for _, row in df.iterrows():
        plt.text(
            row["x"], row["y"],
            str(row["cust_id"]),
            fontsize=10, weight='bold',
            color='black', ha='center', va='center',
            bbox=dict(facecolor='lightgrey', edgecolor='black',
                      boxstyle='circle,pad=0.4', lw=1.0, alpha=1.0),
            zorder=6
        )

    # Plot arrows for each route
    for i, route in enumerate(routes):
        color = colors[i]
        for j in range(len(route) - 1):
            x_start = df.loc[df["cust_id"] == route[j], "x"].values[0]
            y_start = df.loc[df["cust_id"] == route[j], "y"].values[0]
            x_end = df.loc[df["cust_id"] == route[j + 1], "x"].values[0]
            y_end = df.loc[df["cust_id"] == route[j + 1], "y"].values[0]

            arrow = FancyArrowPatch(
                (x_start, y_start), (x_end, y_end),
                arrowstyle='-|>', mutation_scale=20,
                color=color, linewidth=3.0, zorder=3
            )
            plt.gca().add_patch(arrow)

        # Legend entry per route
        plt.plot([], [], color=color, linewidth=3.0, label=f'Route {i + 1}')

    plt.title("CVRPTW Example Routes",
              fontsize=16, weight='bold', pad=20)
    plt.xlabel("X coordinate", fontsize=13)
    plt.ylabel("Y coordinate", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.show()


def solve_cvrptw_from_file(filename: str):
    print(f"=== Solving {filename} ===")

    num_vehicles, capacity, depot, df = read_solomon_file(filename)

    # Example routes
    routes = [
    [0, 47, 48, 43, 0],
    [0, 9, 28, 24, 38, 50, 0],
    [0, 49, 29, 10, 4, 0],
    [0, 40, 39, 7, 32, 31, 34, 0],
    [0, 17, 19, 12, 20, 0],
    [0, 33, 21, 18, 35, 15, 0],
    [0, 23, 30, 11, 0],
    [0, 25, 14, 42, 36, 3, 16, 0],
    [0, 1, 41, 44, 6, 22, 2, 0],
    [0, 5, 0],
    [0, 26, 46, 45, 13, 0],
    [0, 27, 37, 8, 0]
]
    verify_routes(df, routes, capacity)

    #plot_routes(df, routes)


# Example usage
filename = "c100.txt"
solve_cvrptw_from_file(filename)
