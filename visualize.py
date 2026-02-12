import matplotlib.pyplot as plt
import glob

# -------------------------------
# 1️⃣ Parse Solomon R102 file
# -------------------------------
def read_solomon(file_path):
    nodes = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find where CUSTOMER section starts
    start = 0
    for i, line in enumerate(lines):
        if "CUSTOMER" in line:
            start = i + 3   # skip header lines
            break

    for line in lines[start:]:
        if line.strip() == "":
            continue

        parts = line.split()
        if len(parts) < 7:
            continue

        cust_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        demand = float(parts[3])
        ready = float(parts[4])
        due = float(parts[5])
        service = float(parts[6])

        nodes[cust_id] = {
            "x": x,
            "y": y,
            "demand": demand,
            "ready": ready,
            "due": due,
            "service": service
        }

    return nodes


# -------------------------------
# 2️⃣ Load nodes
# -------------------------------
nodes = read_solomon("toy.txt")


# -------------------------------
# 3️⃣ Plot each route snapshot
# -------------------------------
files = sorted(glob.glob("snap/step_*.csv"))

for f in files:
    plt.figure(figsize=(8,8))

    # Plot customers
    for node_id, data in nodes.items():
        if node_id == 0:
            plt.scatter(data["x"], data["y"], s=200, marker='s')
            plt.text(data["x"], data["y"], "Depot")
        else:
            plt.scatter(data["x"], data["y"])
            plt.text(data["x"], data["y"], str(node_id))

    # Draw routes
    with open(f) as file:
        for line in file:
            route = list(map(int, line.strip().split()))

            for i in range(len(route)-1):
                n1 = nodes[route[i]]
                n2 = nodes[route[i+1]]

                plt.plot(
                    [n1["x"], n2["x"]],
                    [n1["y"], n2["y"]]
                )

    plt.title(f)
    plt.show()
