import matplotlib.pyplot as plt
import pandas as pd
import re

def read_vrptw_file(filename):
    """Reads a Solomon VRPTW instance (like C101.txt) and extracts customer data."""
    data_started = False
    customers = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Start reading after 'CUSTOMER' header
            if line.startswith("CUSTOMER"):
                data_started = True
                next(f)  # Skip the column header line
                continue

            if data_started:
                parts = re.split(r'\s+', line)
                if len(parts) >= 8:
                    cust_no = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    demand = float(parts[3])
                    ready = float(parts[4])
                    due = float(parts[5])
                    service = float(parts[6])
                    customers.append((cust_no, x, y, demand, ready, due, service))

    df = pd.DataFrame(customers, columns=["CustNo", "X", "Y", "Demand", "Ready", "Due", "Service"])
    return df

def plot_dataset(df, dataset_name="C101"):
    """Plots customer locations and demand histogram."""
    plt.figure(figsize=(12, 5))

    # --- Customer locations ---
    plt.subplot(1, 2, 1)
    plt.scatter(df["X"], df["Y"], c='brown', s=25)
    plt.scatter(df.loc[df["CustNo"] == 0, "X"], df.loc[df["CustNo"] == 0, "Y"], c='blue', s=40, label="Depot")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(f"Customer Locations: {dataset_name}")
    plt.legend()

    # --- Demand distribution ---
    plt.subplot(1, 2, 2)
    plt.hist(df["Demand"], bins=10, color='dodgerblue', edgecolor='black')
    plt.xlabel("Demand")
    plt.ylabel("Number of Customers")
    plt.title(f"Demand Distribution: {dataset_name}")

    plt.tight_layout()
    plt.show()


# ===== MAIN =====
if __name__ == "__main__":
    file_path = "C101.txt"  # Change to your input file path
    df = read_vrptw_file(file_path)
    print(df.head())  # Verify first few entries
    plot_dataset(df, "C101")
