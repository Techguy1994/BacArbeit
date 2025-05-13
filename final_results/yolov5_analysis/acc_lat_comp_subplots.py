import pandas as pd 
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import os
import numpy as np

def main():
    database = pd.read_csv("database.csv")

    database = database[database['inference type'] != "class"]
    db = database[database['inference type'] != "seg"]

    db = db.drop('top1', axis=1)
    db = db.drop('top5', axis=1)
    db = db.drop('miou', axis=1)
    db = db.drop('inference type', axis=1)
    db = db.drop('os', axis=1)

    db.to_csv("db.csv")



    #database = database.sort_values("latency avg")

    df = db

    df = df.rename(columns={
        "api": "API",
        "model_name": "Model",
        "latency avg": "average latency"
    })

    
    df['API'] = df['API'].replace('onnx', 'Onnx')
    df['API'] = df['API'].replace('ov', 'OpenVINO')
    df['API'] = df['API'].replace('tf', 'Tensorflow')
    df['API'] = df['API'].replace('delegate', 'Armnn Delegate')
    df['API'] = df['API'].replace('pytorch', 'PyTorch')

    df.to_csv("df.csv")

    def facet_plot(data, color, **kwargs):
        # Sort and compute Pareto front
        data_sorted = data.sort_values(by="average latency")
        front = []
        max_map = -float("inf")
        for _, row in data_sorted.iterrows():
            if row["map"] > max_map:
                front.append(row)
                max_map = row["map"]
        pareto = pd.DataFrame(front)

        # Plot scatter and Pareto line
        sns.scatterplot(data=data, x="map", y="average latency", hue="API", s=100, **kwargs)
        sns.lineplot(data=pareto, x="map", y="average latency", color="black", linestyle="--", **kwargs)

    # Create FacetGrid
    g = sns.FacetGrid(df, col="Model", col_wrap=2, height=4.5, aspect=1.2, sharex=False, sharey=False)
    g.map_dataframe(facet_plot)

    # Add labels and legend
    g.set_titles("{col_name}")
    g.set_axis_labels("mAP", "Average Latency ms")
    g.add_legend(title="Framework")
    g.fig.subplots_adjust(top=0.9)
    #g.fig.suptitle("Latency vs mAP per YOLOv5 Model with Pareto Front", fontsize=16)

    handles, labels = g.axes[0].get_legend_handles_labels()
    g._legend.remove()
    g.fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.05, 0.98), title="Framework")

    plt.tight_layout()
    plt.savefig("detection_scatter.pdf", bbox_inches='tight')
    #plt.savefig("bar_plot_4_subplot.png", dpi=300, bbox_inches='tight')




if __name__ == "__main__":
    main()