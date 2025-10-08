import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors


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

    plt.title("CVRPTW Routes Visualization (cuOpt - Toy.txt)",
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
        [0, 5, 4, 0],
        [0, 2, 6, 7, 0],
        [0, 1, 3, 8,9,10, 0]
    ]

    plot_routes(df, routes)


# Example usage
filename = "toy.txt"
solve_cvrptw_from_file(filename)
