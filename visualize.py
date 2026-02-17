import matplotlib.pyplot as plt
import glob
import re

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
        nodes[cust_id] = {
            "x": float(parts[1]),
            "y": float(parts[2]),
            "demand": float(parts[3]),
            "ready": float(parts[4]),
            "due": float(parts[5]),
            "service": float(parts[6])
        }

    return nodes


# -------------------------------
# 2️⃣ Load nodes
# -------------------------------
nodes = read_solomon("c_50.txt")

# -------------------------------
# 3️⃣ Load snapshot files
# -------------------------------
files = glob.glob("snap/step_*.csv")

# Sort by numeric step value
files = sorted(
    files,
    key=lambda x: int(re.search(r'step_(\d+)', x).group(1))
)
print(files)
current_index = 0

fig, ax = plt.subplots(figsize=(8, 8))


def draw_snapshot(index):
    ax.clear()

    # Plot nodes
    for node_id, data in nodes.items():
        if node_id == 0:
            ax.scatter(data["x"], data["y"], s=200, marker='s')
            ax.text(data["x"], data["y"], "Depot")
        else:
            ax.scatter(data["x"], data["y"])
            ax.text(data["x"], data["y"], str(node_id))

    # Draw routes
    with open(files[index]) as file:
        for line in file:
            route = list(map(int, line.strip().split()))

            # Add depot at start and end
            route = [0] + route + [0]

            for i in range(len(route) - 1):
                n1 = nodes[route[i]]
                n2 = nodes[route[i+1]]

                ax.plot(
                    [n1["x"], n2["x"]],
                    [n1["y"], n2["y"]]
                )

    ax.set_title(f"Snapshot {index+1} / {len(files)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    fig.canvas.draw()


def on_key(event):
    global current_index

    if event.key == 'right':
        current_index = (current_index + 1) % len(files)
        draw_snapshot(current_index)

    elif event.key == 'left':
        current_index = (current_index - 1) % len(files)
        draw_snapshot(current_index)


fig.canvas.mpl_connect('key_press_event', on_key)

# Draw first snapshot
draw_snapshot(current_index)

plt.show()
