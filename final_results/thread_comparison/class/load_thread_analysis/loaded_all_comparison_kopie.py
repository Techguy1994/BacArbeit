import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd 
import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    database = pd.read_csv("load_database_updated.csv")
    df = create_empty_dataframe()

    db = database[database['Frameworks'] != "PyTorch"]

    unique_apis = db['Frameworks'].unique()
    unique_loads = db["Load"].unique()

    print(unique_apis, unique_loads)
    

    db = db.sort_values('Load')

    load_order = ['No Load', 'One Core Load', 'Two Core Load', 'Three Core Load']
    db['Load'] = pd.Categorical(db['Load'], categories=load_order, ordered=True)

    
    # 2. Fix the 'thread count' order
    thread_order = ['Default', 'One Core', 'Two Cores', 'Three Cores', 'Four Cores']
    db['Core Count'] = pd.Categorical(db['Core Count'], categories=thread_order, ordered=True)


    db.to_csv("bar.csv")

    print(unique_apis)
    
    # Create the bar plot without automatic error bars
    g = sns.catplot(
        data=db,
        kind="bar",
        x="Load",
        y="Median Latency [s]",
        hue="Core Count",
        col="Frameworks",
        col_wrap=2,
        palette="viridis",
        errorbar=None  # Disable automatic error bars
    )

    # Set consistent y-axis limits
    for ax in g.axes.flat:
        ax.set_ylim(0, 0.4)

    # Set titles and labels
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_title(ax.get_title(), fontsize=20)
    g.set_axis_labels("Artificial Load", "Median Latency (s)", fontsize=16)
    g._legend.set_title("Core counts used for inference")
    g._legend.set_bbox_to_anchor((0.06, 1.0))
    g._legend.set_loc("upper left")

    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    # Set font size for legend
    for text in g._legend.get_texts():
        text.set_fontsize(13)
    g._legend.get_title().set_fontsize(15)

    # Iterate over axes and their corresponding facet value (i.e., 'Framework')
    for ax, framework in zip(g.axes.flat, db["Frameworks"].unique()):
        # Filter data for this specific facet
        sub_db = db[db["Frameworks"] == framework]

        # Get bar containers
        for container in ax.containers:
            for bar in container:
                # Get bar position and height
                x = bar.get_x() + bar.get_width() / 2
                height = bar.get_height()

                # Retrieve bar info using x and hue label
                load = bar.get_label()
                core_count = container.get_label()

                # Locate the corresponding row in your DataFrame
                row = sub_db[
                    (sub_db["Load"] == load) &
                    (sub_db["Core Count"] == core_count)
                ]
                if not row.empty:
                    err = row["Standard Deviation [s]"].values[0]
                    ax.errorbar(
                        x, height, yerr=err,
                        fmt='none', color='black', capsize=4, linewidth=1
                    )

    # Save the plot
    plt.tight_layout()
    plt.savefig("bar_plot_4_subplot.pdf", bbox_inches='tight')
        #plt.savefig("bar_plot_4_subplot.png", dpi=300, bbox_inches='tight')


        


def create_empty_dataframe():
    dict = {
    "mean latency": [],
    "thread": [],
    "api": [],
    "load": []
}

    df = pd.DataFrame(dict)

    return df
            







if __name__ == "__main__":
    main()
