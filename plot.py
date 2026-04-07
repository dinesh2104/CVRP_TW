import pandas as pd
import matplotlib.pyplot as plt

# Data preparation - Fixed missing comma in Algorithm list
data = {
    "Algorithm": [
        "cuOpt", 
        "CVRPTW_SeqMDS",
        "CVRPTW_parMDS",
        "C&W (no post-optim)", 
        "Sweep+C&W (Seq)", 
        "Sweep+C&W (Par)"
    ],
    "Distance (Cost)": [35382.62, 120682.625, 120913.9333, 47050.11667, 38760.7583, 38745.1083],
    "Total Time": [61.2, 43.74481667, 135.2070833, 351.43, 15.11, 2.74],
    "Vehicles Used": [58.92, 290.75, 291.5833333, 78.58, 70.5, 69.83],
}

df = pd.DataFrame(data)

# Modern professional color palette (6 colors to match 6 algorithms)
colors = ['#264653', '#5a6735', '#6a34a7', '#2a9d8f', '#e9c46a', '#f4a261']

metrics = ["Distance (Cost)", "Total Time", "Vehicles Used"]

# Generate separate graphs for each metric
for metric in metrics:
    plt.figure(figsize=(12, 7))
    bars = plt.bar(df['Algorithm'], df[metric], color=colors)
    
    # Adding titles and labels
    plt.title(f'{metric} Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=20, ha='right') # Rotated for better readability
    
    # Adding data labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.01), 
                 f'{yval:,.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    # To save: plt.savefig(f"{metric.lower().replace(' ', '_')}.png") 
    plt.show()