import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def read_vrp_file(filename: str):
    """
    Reads a TSPLIB-style CVRP instance file.
    """
    with open(filename, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    # Extract capacity
    cap_line = [l for l in lines if l.startswith("CAPACITY")][0]
    capacity = int(cap_line.split(":")[1].strip())

    # Find section indices
    node_idx = lines.index("NODE_COORD_SECTION") + 1
    demand_idx = lines.index("DEMAND_SECTION") + 1
    depot_idx = lines.index("DEPOT_SECTION") + 1
    eof_idx = lines.index("EOF")

    # Parse node coordinates
    node_data = []
    for line in lines[node_idx:demand_idx - 1]:
        parts = line.split()
        if len(parts) >= 3:
            cid = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            node_data.append((cid, x, y))
    node_df = pd.DataFrame(node_data, columns=["cust_id", "x", "y"])

    # Parse demand
    demand_data = []
    for line in lines[demand_idx:depot_idx - 1]:
        parts = line.split()
        if len(parts) >= 2:
            cid = int(parts[0])
            demand = int(parts[1])
            demand_data.append((cid, demand))
    demand_df = pd.DataFrame(demand_data, columns=["cust_id", "demand"])

    # Merge data
    df = pd.merge(node_df, demand_df, on="cust_id", how="left")

    # Parse depot
    depot = int(lines[depot_idx].split()[0])

    return capacity, depot, df


def plot_routes(df, routes):
    plt.figure(figsize=(10, 8))

    color_list = [
        "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080"
    ]
    colors = color_list[:len(routes)]

    # Plot depot
    depot = df.loc[df["cust_id"] == 0]
    depot_x, depot_y = depot["x"].values[0], depot["y"].values[0]
    plt.scatter(depot_x, depot_y, c='red', s=250, marker='s',
                edgecolor='black', label='Depot', zorder=5)

    # Plot customers
    cust_df = df[df["cust_id"] != 0]
    plt.scatter(cust_df["x"], cust_df["y"],
                c='lightgrey', s=180, edgecolor='black', linewidth=1.2,
                label='Customers', zorder=4)

    # Add labels
    for _, row in df.iterrows():
        plt.text(
            row["x"], row["y"] + 0.3, str(int(row["cust_id"])),
            fontsize=10, ha='center', va='bottom', weight='bold'
        )

    # Draw routes
    for i, route in enumerate(routes):
        color = colors[i]
        for j in range(len(route) - 1):
            x1 = df.loc[df["cust_id"] == route[j], "x"].values[0]
            y1 = df.loc[df["cust_id"] == route[j], "y"].values[0]
            x2 = df.loc[df["cust_id"] == route[j + 1], "x"].values[0]
            y2 = df.loc[df["cust_id"] == route[j + 1], "y"].values[0]

            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='-|>', mutation_scale=20,
                color=color, linewidth=3, zorder=2)
            plt.gca().add_patch(arrow)

        plt.plot([], [], color=color, linewidth=3, label=f"Route {i + 1}")

    plt.title("CVRP Example Routes", fontsize=16, pad=20)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


def solve_cvrp_from_file(filename: str):
    print(f"=== Solving {filename} ===")

    capacity, depot, df = read_vrp_file(filename)

    # Example routes (you can modify these)
    routes = [
        [0, 1, 4, 0],
        [0, 2, 5, 6, 3, 0]
    ]

    print(f"Vehicle Capacity: {capacity}")
    print(f"Depot Node: {depot}")
    print(df)
    plot_routes(df, routes)


# Example usage
filename = "ch2_data2.txt"
solve_cvrp_from_file(filename)
