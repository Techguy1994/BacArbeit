import pandas as pd 
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import os
import numpy as np

def main():
    df = pd.read_csv("database_updated.csv")

    df = df[df['inference type'] != "class"]
    df = df[df['inference type'] != "seg"]

    df = df.drop('Top1', axis=1)
    df = df.drop('Top5', axis=1)
    df = df.drop('mIoU', axis=1)
    df = df.drop('inference type', axis=1)

    df.to_csv("df.csv")

    def facet_plot(data, color, **kwargs):
        # Sort and compute Pareto front
        data_sorted = data.sort_values(by="Median Latency [s]")
        front = []
        max_map = -float("inf")
        for _, row in data_sorted.iterrows():
            if row["mAP"] > max_map:
                front.append(row)
                max_map = row["mAP"]
        pareto = pd.DataFrame(front)

        # Plot scatter and Pareto line
        sns.scatterplot(data=data, x="mAP", y="Median Latency [s]", hue="Frameworks", s=300, **kwargs)
        sns.lineplot(data=pareto, x="mAP", y="Median Latency [s]", color="black", linestyle="--", **kwargs, linewidth = 2)
        plt.grid(True)

    # Create FacetGrid
    g = sns.FacetGrid(df, col="Model", col_wrap=2, height=4.5, aspect=1.2, sharex=False, sharey=False)
    g.map_dataframe(facet_plot)

    for ax in g.axes.flatten():
        ax.tick_params(axis='both', labelsize=15)

    # Add labels and legend
    g.set_titles("{col_name}", size=24)
    g.set_axis_labels("mAP", "Median Latency [s]", fontsize=18)
    g.add_legend(title="Framework")
    g.fig.subplots_adjust(top=0.9)
    #g.fig.suptitle("Latency vs mAP per YOLOv5 Model with Pareto Front", fontsize=16)

    handles, labels = g.axes[0].get_legend_handles_labels()
    g._legend.remove()
    g.fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.55, 0.95), title="Framework", fontsize=14, title_fontsize=16)

    plt.tight_layout()
    plt.grid(True)
    plt.savefig("detection_scatter.pdf", bbox_inches='tight')
    #plt.savefig("bar_plot_4_subplot.png", dpi=300, bbox_inches='tight')




if __name__ == "__main__":
    main()