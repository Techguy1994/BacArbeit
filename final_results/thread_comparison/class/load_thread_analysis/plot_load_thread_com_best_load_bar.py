import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd 
import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    database = pd.read_csv("load_database.csv")
    df = create_empty_dataframe()

    db = database[database['api'] != "pytorch"]

    unique_apis = db['api'].unique()
    unique_loads = db["load"].unique()

    sort_loads = ["default", "one", "two", "three"]

    db = db.sort_values('load')

    load_order = ['default', 'one', 'two', 'three']
    db['load'] = pd.Categorical(db['load'], categories=load_order, ordered=True)

    # 2. Fix the 'thread count' order
    thread_order = ['False', '1', '2', '3', '4']
    db['thread count'] = pd.Categorical(db['thread count'], categories=thread_order, ordered=True)

    load_mapping = {
    'default': 'no load',
    'one': '1 thread load',
    'two': '2 thread load',
    'three': '3 thread load'
}
    
    api_mapping = {
        "ov": "OpenVINO",
        "onnx": "Onnx",
        "tf": "Tensorflow",
        "delegate": "ArmNN Delegate"
    }

    # Apply the mapping
    db["api"] = db["api"].replace(api_mapping)
    db['load'] = db['load'].replace(load_mapping)
    db['thread count'] = db['thread count'].replace({"False": 'Default'})

    db.to_csv("bar.csv")

    print(unique_apis)
    
    g = sns.catplot(
        data=db, kind="bar",
        x="load", y="latency avg", hue="thread count",
        col="api", col_wrap=2,
        palette="viridis"
    )

    for ax in g.axes.flat:
        ax.set_ylim(0, 0.4)

    #g.add_legend(title="Thread Count", bbox_to_anchor=(1.05, 0.5), loc="center left")

    # Optional: rename axes or titles
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_title(ax.get_title(), fontsize=20)
    g.set_axis_labels("Artificial Load", "Latency (s)", fontsize = 16)
    g._legend.set_title("Thread Count")
    g._legend.set_bbox_to_anchor((0.06, 1.0)) 
    g._legend.set_loc("upper left")  

    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    # Set font size for legend labels
    for text in g._legend.get_texts():
        text.set_fontsize(13)

    # Optionally adjust the legend title font size too
    g._legend.get_title().set_fontsize(15)

    plt.tight_layout()
    plt.savefig("bar_plot_4_subplot.pdf", bbox_inches='tight')
    plt.savefig("bar_plot_4_subplot.png", dpi=300, bbox_inches='tight')


        


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
