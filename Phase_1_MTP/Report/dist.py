import matplotlib.pyplot as plt
import pandas as pd
import re

def read_solomon_file(filename: str):
    """
    Parse Solomon VRPTW instance file (like C101.txt).
    Returns depot index, vehicle count, vehicle capacity,
    location coordinates, demands, time windows, and service times.
    """
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

    df = pd.DataFrame(cust_data,
                      columns=["cust_id", "x", "y", "demand", "ready", "due", "service"])

    depot = 0 

    return df

def plot_dataset(df, dataset_name="C101"):
    """Plots customer locations and demand histogram."""
    # plt.figure(figsize=(12, 5))

    # --- Customer locations ---
    #plt.subplot(1, 2, 1)
    plt.scatter(df["x"], df["y"], c='brown', s=25)
    plt.scatter(df.loc[df["cust_id"] == 0, "x"], df.loc[df["cust_id"] == 0, "y"], c='blue', s=40, label="Depot")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(f"Customer Locations: {dataset_name}")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ===== MAIN =====
if __name__ == "__main__":
    file_path = "rc203.txt"  # Change to your input file path
    df = read_solomon_file(file_path)
    print(df.head())  # Verify first few entries
    plot_dataset(df, "rc203")
