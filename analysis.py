import pandas as pd
import numpy as np
import scipy.spatial.distance as distance


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

    df = pd.DataFrame(
        cust_data,
        columns=["cust_id", "x", "y", "demand", "ready", "due", "service"]
    )

    depot = 0  # always first entry in Solomon format

    return num_vehicles, capacity, depot, df


def solve_cvrptw_from_file(filename: str):
    """
    Solve CVRPTW from Solomon benchmark file using cuOpt.
    """
    print(f"=== Solving {filename} ===")

    # Read data
    num_vehicles, capacity, depot, df = read_solomon_file(filename)

    # Cost matrix
    coords = df[["x", "y"]].to_numpy()
    dist_matrix = distance.cdist(coords, coords, metric="euclidean")

    # Round for readability
    dist_matrix = np.round(dist_matrix, 2)

    # Build labeled DataFrame
    labels = df["cust_id"].tolist()
    dist_matrix_df = pd.DataFrame(dist_matrix, index=labels, columns=labels)

    # Print neatly
    print("\n=== Distance Matrix ===")
    print(dist_matrix_df.to_string())

    print(" \n ====== Time taken to travel between locations (in minutes) ====== \n")
    time_matrix = dist_matrix / 50 * 60  # Convert to minutes
    time_matrix_df = pd.DataFrame(time_matrix, index=labels, columns=labels)
    print(time_matrix_df.to_string())


if __name__ == "__main__":
    solve_cvrptw_from_file("toy.txt")
