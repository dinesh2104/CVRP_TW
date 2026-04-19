# import pandas as pd
# import matplotlib.pyplot as plt

# # Data preparation - Fixed missing comma in Algorithm list
# data = {
#     "Algorithm": [
#         "cuOpt", 
#         "CVRPTW_SeqMDS",
#         "CVRPTW_parMDS",
#         "C&W (no post-optim)", 
#         "Sweep+C&W (Seq)", 
#         "Sweep+C&W (Par)"
#     ],
#     "Distance (Cost)": [35382.62, 120682.625, 120913.9333, 47050.11667, 38760.7583, 38745.1083],
#     "Total Time": [61.2, 43.74481667, 135.2070833, 351.43, 15.11, 2.74],
#     "Vehicles Used": [58.92, 290.75, 291.5833333, 78.58, 70.5, 69.83],
# }

# df = pd.DataFrame(data)

# # Modern professional color palette (6 colors to match 6 algorithms)
# colors = ['#264653', '#5a6735', '#6a34a7', '#2a9d8f', '#e9c46a', '#f4a261']

# metrics = ["Distance (Cost)", "Total Time", "Vehicles Used"]

# # Generate separate graphs for each metric
# for metric in metrics:
#     plt.figure(figsize=(12, 7))
#     bars = plt.bar(df['Algorithm'], df[metric], color=colors)
    
#     # Adding titles and labels
#     plt.title(f'{metric} Comparison', fontsize=14, fontweight='bold', pad=15)
#     plt.ylabel(metric, fontsize=12)
#     plt.xticks(rotation=20, ha='right') # Rotated for better readability
    
#     # Adding data labels on top of each bar
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.01), 
#                  f'{yval:,.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
#     plt.grid(axis='y', linestyle='--', alpha=0.3)
#     plt.tight_layout()
#     # To save: plt.savefig(f"{metric.lower().replace(' ', '_')}.png") 
#     plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Exact color codes from the provided image
# IJPP_BLUE = '#9999FF'  # Light blue for boxes
# IJPP_RED = '#FF9999'   # Light red/pink for boxes
# OUTLINE_BLUE = '#0000FF'
# OUTLINE_RED = '#FF0000'

# phases = ['Pre-Processing', 'Route Construction', 'Post-Processing']
# sequential_times = [0.0002199776806, 25.58412028, 5.669941635]
# parallel_times = [0.0002132468667, 2.173925298, 0.3318464733]

# x = np.arange(len(phases))
# width = 0.35

# fig, ax = plt.subplots(figsize=(8, 5))
# ax.set_yscale('log')

# # Applying the colors from the image
# rects1 = ax.bar(x - width/2, sequential_times, width, label='Sequential', 
#                 color=IJPP_BLUE, edgecolor=OUTLINE_BLUE, linewidth=1)
# rects2 = ax.bar(x + width/2, parallel_times, width, label='Parallel', 
#                 color=IJPP_RED, edgecolor=OUTLINE_RED, linewidth=1)

# ax.set_ylabel('Execution Time (s)')
# ax.set_xticks(x)
# ax.set_xticklabels(phases)
# ax.legend()

# # Match the dashed grid from your image
# ax.grid(True, linestyle='--', alpha=0.6)

# # plt.savefig('comparison_plot.eps', format='eps')
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# solvers = ['BKS', 'cuOpt', 'CVRPTW_SOLVER(sequential)', 'CVRPTW_SOLVER(parallel)']
# avg_distances = [33666.30, 36440.301, 38007.84, 38047.67]

# # Professional Color Palette
# # Using the soft blue and coral red shades previously requested
# colors = ['#B0C4DE', '#87CEFA', '#9999FF', '#FF9999']

# fig, ax = plt.subplots(figsize=(9, 6))
# bars = ax.bar(solvers, avg_distances, color=colors, edgecolor='black', width=0.6)

# # Adding value labels on top of bars
# for bar in bars:
#     height = bar.get_height()
#     ax.annotate(f'{height:,.1f}',
#                 xy=(bar.get_x() + bar.get_width() / 2, height),
#                 xytext=(0, 5),  # 5 points vertical offset
#                 textcoords="offset points",
#                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# # Academic Styling
# ax.set_ylabel('Average Distance', fontsize=12, fontweight='bold')
# ax.set_title('Solver Performance Comparison: Solution Quality', fontsize=14, fontweight='bold', pad=20)
# ax.set_ylim(0, 45000)

# # Removing unnecessary spines for a clean "Nature-style" look
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.grid(axis='y', linestyle='--', alpha=0.5)

# plt.tight_layout()
# # Saving in vector format for LaTeX
# # plt.savefig('distance_comparison.eps', format='eps', dpi=300)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data Extraction
categories = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
bks = [41671.34, 16772.96, 46901.26, 28842.18, 43895.78, 23914.28]
cuopt = [44213.46, 18458.05, 51207.58, 32163.87, 47367.30, 25231.548]
parallel = [44397.46, 19568.14, 53071.06, 33954.6, 49326.47, 27968.3]
sequential = [44401.71, 19495.29, 53084.64, 33845.3, 49218.76, 28001.34]

x = np.arange(len(categories))
width = 0.2  # width of the bars

# Color Palette (consistent with IJPP standards)
color_bks = '#D3D3D3'       # Light Gray
color_cuopt = '#87CEFA'     # Light Sky Blue
color_parallel = '#9999FF'   # Soft Blue
color_sequential = '#FF9999' # Coral Red

fig, ax = plt.subplots(figsize=(12, 7))

# Create grouped bars
ax.bar(x - 1.5*width, bks, width, label='BKS', color=color_bks, edgecolor='black')
ax.bar(x - 0.5*width, cuopt, width, label='cuOpt', color=color_cuopt, edgecolor='black')
ax.bar(x + 0.5*width, parallel, width, label='Parallel Solver', color=color_parallel, edgecolor='black')
ax.bar(x + 1.5*width, sequential, width, label='Sequential Solver', color=color_sequential, edgecolor='black')

# Labeling and Formatting
ax.set_ylabel('Total Distance', fontsize=12, fontweight='bold')
ax.set_title('Solver Performance Comparison by Category (1000-Node Solomon)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.legend(fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
# plt.savefig('category_comparison.eps', format='eps', dpi=300)
plt.show()