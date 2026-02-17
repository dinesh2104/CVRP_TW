import math
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Parse Solomon file
# -------------------------------
def read_solomon(file_path):
    nodes = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    start = 0
    for i, line in enumerate(lines):
        if "CUSTOMER" in line:
            start = i + 3
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
# 2️⃣ Euclidean distance
# -------------------------------
def distance(a, b):
    return math.sqrt((a["x"] - b["x"])**2 +
                     (a["y"] - b["y"])**2)


# -------------------------------
# 3️⃣ Compute Clarke-Wright savings
# S_ij = d(0,i) + d(j,0) - d(i,j)
# -------------------------------
def compute_savings(nodes):
    savings = {}

    depot = nodes[0]

    for i in nodes:
        if i == 0:
            continue

        for j in nodes:
            if j == 0 or i == j:
                continue

            d0i = distance(depot, nodes[i])
            dj0 = distance(nodes[j], depot)
            dij = distance(nodes[i], nodes[j])

            saving = d0i + dj0 - 5*dij

            savings[(i, j)] = saving

    return savings


# -------------------------------
# 4️⃣ Load nodes
# -------------------------------
nodes = read_solomon("c_50.txt")

# -------------------------------
# 5️⃣ Calculate savings
# -------------------------------
savings = compute_savings(nodes)

# Print top 10 savings
sorted_savings = sorted(savings.items(),
                        key=lambda x: x[1],
                        reverse=True)



i1=21
j1=24

print("Euclidean distance savings for pair (i,j):")

print(f"d({i1},{j1}) = {distance(nodes[i1], nodes[j1]):.2f}")

print("Saving values for pairs (i,j)")
for (i, j), s in sorted_savings:
    if (i == i1 and j == j1) or (i == j1 and j == i1):
        print(f"S({i},{j}) = {s:.2f}")