import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------
# Data
# ----------------------------------
customers = [100, 200, 400, 800, 1000]

# Average Total Cost
cuopt_cost = [1003.33, 3000.3625, 6686.088333, 23647.26417, 35382.62167]
seq_cost   = [1815.986, 7249.005833, 17563.50583, 75572.6, 120682.625]
par_cost   = [1823.052667, 7257.876667, 17628.275, 75742.725, 120913.9333]

# Average Execution Time (in seconds)
cuopt_time = [57.81583333, 58.96166667, 60.51666667, 60.8675, 61.20916667]
seq_time   = [4.31389, 8.857429167, 16.904675, 34.57569167, 43.74481667]
par_time   = [17.85458333, 31.155075, 56.57726667, 108.8358333, 135.2070833]

# ----------------------------------
# Plot 1: Average Total Cost vs Number of Customers
# ----------------------------------
plt.figure(figsize=(9,6))
plt.plot(customers, seq_cost, marker='o', label="Sequential", linewidth=2)
plt.plot(customers, par_cost, marker='s', label="Parallel", linewidth=2)
plt.plot(customers, cuopt_cost, marker='^', label="cuOpt", linewidth=2)

plt.title("Average Total Cost vs Number of Customers", fontsize=14, fontweight='bold')
plt.xlabel("Number of Customers", fontsize=12)
plt.ylabel("Average Total Cost", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# ----------------------------------
# Plot 2: Execution Time vs Number of Customers
# ----------------------------------
plt.figure(figsize=(9,6))
plt.plot(customers, seq_time, marker='o', label="Sequential", linewidth=2)
plt.plot(customers, par_time, marker='s', label="Parallel", linewidth=2)
plt.plot(customers, cuopt_time, marker='^', label="cuOpt", linewidth=2)

plt.title("Average Execution Time vs Number of Customers", fontsize=14, fontweight='bold')
plt.xlabel("Number of Customers", fontsize=12)
plt.ylabel("Execution Time (seconds)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
