import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
df = pd.read_csv("database_updated.csv")
df = df[~df['inference type'].isin(["seg", "det"])]
df = df.drop(columns=['mIoU', 'mAP', 'inference type'])

# Configuration: (Model, Metric)
plots = [
    ("MobileNetV3 Small", "Top1"),
    ("MobileNetV3 Small", "Top5"),
    ("MobileNetV3 Large", "Top5"),
    ("MobileNetV2", "Top5"),
]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

# Collect handles for legends
all_framework_handles = {}
all_model_handles = {}

for i, (model, top) in enumerate(plots):
    ax = axes[i]
    df_model = df[df["Model"].str.contains(model)].copy()

    # Drop unused metric column
    if top == "Top1" and "Top5" in df_model.columns:
        df_model = df_model.drop("Top5", axis=1)
    elif top == "Top5" and "Top1" in df_model.columns:
        df_model = df_model.drop("Top1", axis=1)

    df_model = df_model.sort_values(by='Median Latency [s]')

    # Pareto front extraction
    pareto_front = []
    max_top = -float("inf")
    for _, row in df_model.iterrows():
        if row[top] > max_top:
            pareto_front.append(row)
            max_top = row[top]
    pareto_df = pd.DataFrame(pareto_front)

    # Scatterplot (captures legend handles)
    sns.scatterplot(
        data=df_model,
        x=top,
        y="Median Latency [s]",
        hue="Frameworks",
        style="Model",
        s=450,
        ax=ax
    )

    # Pareto front line
    sns.lineplot(
        data=pareto_df,
        x=top,
        y="Median Latency [s]",
        color="black",
        linewidth=2,
        marker="o",
        label="Pareto Front",
        linestyle="--",
        alpha=1.0,
        markersize=8,
        ax=ax
    )

    ax.set_title(f"{model} - {top}", fontsize=26)
    ax.set_xlabel(top, fontsize=24)
    ax.set_ylabel("Median Latency [s]", fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True)

    # Collect legend handles
    handles, labels = ax.get_legend_handles_labels()
    for h, l in zip(handles, labels):
        if l in df_model["Frameworks"].unique():
            all_framework_handles[l] = h
        elif l in df_model["Model"].unique():
            all_model_handles[l] = h

# Remove auto legends from subplots
for ax in axes:
    ax.legend_.remove()

# Place manual legends
framework_legend = fig.legend(
    handles=list(all_framework_handles.values()),
    labels=list(all_framework_handles.keys()),
    title="Frameworks",
    loc='upper center',
    bbox_to_anchor=(0.2, 1.05),
    fontsize=17,
    title_fontsize=20,
    ncol=3
)

model_legend = fig.legend(
    handles=list(all_model_handles.values()),
    labels=list(all_model_handles.keys()),
    title="Model",
    loc='upper center',
    bbox_to_anchor=(0.71, 1.05),
    fontsize=17,
    title_fontsize=20,
    ncol=3
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("class_scatter_subplots.pdf", bbox_inches='tight')
#plt.show()
