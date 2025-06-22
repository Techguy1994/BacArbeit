import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from matplotlib.lines import Line2D

df = pd.read_csv("database_updated.csv")
df = df[df['inference type'] != "class"]
df = df[df['inference type'] != "det"]
df = df.drop(columns=['Top1', 'Top5', 'mAP', 'inference type'])
df["mIoU"] = df["mIoU"] / 100
df = df.sort_values(by='Median Latency [s]')

# Extract Pareto front
pareto_front = []
max_miou = -float("inf")
for _, row in df.iterrows():
    if row["mIoU"] > max_miou:
        pareto_front.append(row)
        max_miou = row["mIoU"]
pareto_df = pd.DataFrame(pareto_front)

# Assign a unique color per framework for consistency
unique_frameworks = df["Frameworks"].unique()
palette = sns.color_palette("tab10", n_colors=len(unique_frameworks))
color_map = dict(zip(unique_frameworks, palette))

# Create the plot
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=df,
    x="mIoU",
    y="Median Latency [s]",
    hue="Frameworks",
    style="Model",
    s=200,
    palette=color_map,
    legend=False  # We'll use a custom legend
)

# Plot Pareto front
sns.lineplot(
    data=pareto_df,
    x="mIoU",
    y="Median Latency [s]",
    color="black",
    linewidth=2,
    marker="o",
    label="Pareto Front",
    alpha=0.5,
    linestyle="--"
)

# Sort and build custom legend handles
df_sorted = df.sort_values(by="Frameworks")

# Build custom legend handles
custom_handles = [
    Line2D(
        [], [], linestyle="none", marker="o",
        markersize=10,
        label=f"{row['Frameworks']} - {row['Model']} ({row['Median Latency [s]']:.2f}s)",
        color=color_map[row["Frameworks"]]
    )
    for _, row in df_sorted.iterrows()
]

# Add Pareto Front as the LAST item
custom_handles.append(
    Line2D([], [], color="black", linestyle="--", marker="o", label="Pareto Front")
)

# Custom legend on the right
plt.legend(
    handles=custom_handles,
    title="Framework - Model (Latency)",
    loc="upper left",
    bbox_to_anchor=(1.01, 1.01),
    fontsize=10,
    title_fontsize=12,
    labelspacing=1.2
)

plt.title("DeeplabV3")
plt.xlabel("mIoU")
plt.ylabel("Median Latency [s]")
plt.tight_layout()
plt.grid(True)
plt.savefig("deeplab_scatter.pdf", bbox_inches='tight')
#plt.show()


