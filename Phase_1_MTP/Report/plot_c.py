import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------
# Data
# ----------------------------------
methods = ['Sequential', 'Parallel', 'cuOpt']

# Average Cost values
avg_cost = [44576.74, 44673.17, 13943.93]

# Average Execution Time values (in seconds)
avg_time = [21.68, 69.33, 59.87]

# Colors (from your 3rd screenshot)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# ----------------------------------
# Plot 1: Average Cost Comparison
# ----------------------------------
plt.figure(figsize=(8,5))
bars = plt.bar(methods, avg_cost, color=colors, width=0.6)

# Add numeric labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 500, f"{yval:,.2f}", 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("Average Cost Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Average Total Cost", fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ----------------------------------
# Plot 2: Average Execution Time Comparison
# ----------------------------------
plt.figure(figsize=(8,5))
bars = plt.bar(methods, avg_time, color=colors, width=0.6)

# Add numeric labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}s", 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("Average Execution Time Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Execution Time (seconds)", fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
