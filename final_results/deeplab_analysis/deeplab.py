import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Load and preprocess
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

# Color and marker maps
frameworks = df["Frameworks"].unique()
models = df["Model"].unique()
palette = sns.color_palette("tab10", len(frameworks))
color_map = dict(zip(frameworks, palette))

markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '<', '>', '*']
marker_map = dict(zip(models, markers))

# Set up plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="mIoU",
    y="Median Latency [s]",
    hue="Frameworks",
    style="Model",
    palette=color_map,
    markers=marker_map,
    s=200,
    legend=False,
    ax=ax
)

# Pareto front line
sns.lineplot(
    data=pareto_df,
    x="mIoU",
    y="Median Latency [s]",
    color="black",
    linewidth=2,
    marker="o",
    alpha=1,
    linestyle="--",
    ax=ax
)

# Custom legend handles
framework_handles = [
    Line2D([], [], marker='o', linestyle='none', color=color_map[fw], label=fw, markersize=10)
    for fw in sorted(frameworks)
]

model_handles = [
    Line2D([], [], marker=marker_map[m], linestyle='none', color='gray', label=m, markersize=10)
    for m in sorted(models)
]

pareto_handle = Line2D([], [], color="black", linestyle="--", marker="o", label="Pareto Front", markersize=10)

# Add legends
legend1 = ax.legend(
    handles=framework_handles,
    title="Frameworks",
    loc='upper left',
    bbox_to_anchor=(1.01, 1.02),
    title_fontsize=16,
    fontsize=14
)
ax.add_artist(legend1)

legend2 = ax.legend(
    handles=model_handles,
    title="Models",
    loc='upper left',
    bbox_to_anchor=(1.01, 0.6),
    title_fontsize=16,
    fontsize=14
)
ax.add_artist(legend2)

#legend3 = ax.legend(
#    handles=[pareto_handle],
#    loc='upper left',
#    bbox_to_anchor=(1.01, 0.3),
#    title=None,
#    fontsize=14
#)
#ax.add_artist(legend3)

# Axis labels and title
ax.set_title("DeeplabV3", fontsize=24)
ax.set_xlabel("mIoU", fontsize=20)
ax.set_ylabel("Median Latency [s]", fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Save directly to PDF — include external legends
fig.savefig("deeplab_scatter.pdf", bbox_extra_artists=(legend1,legend2), bbox_inches='tight')
