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

    sys.exit()


    db.to_csv("bar.csv")

    print(unique_apis)
    
    g = sns.catplot(
        data=db, kind="bar",
        x="Load", y="Median Latency [s]", hue="Core Count",
        col="Frameworks", col_wrap=2,
        palette="viridis"
    )

    for ax in g.axes.flat:
        ax.set_ylim(0, 0.4)

    #g.add_legend(title="Thread Count", bbox_to_anchor=(1.05, 0.5), loc="center left")

    # Optional: rename axes or titles
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_title(ax.get_title(), fontsize=20)
    g.set_axis_labels("Artificial Load", "Median Latency (s)", fontsize = 16)
    g._legend.set_title("Core counts used for infernce")
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
    #plt.savefig("bar_plot_4_subplot.png", dpi=300, bbox_inches='tight')


        


def create_empty_dataframe():
    dict = {
    "Latency []": [],
    "Core Count": [],
    "Frameworks": [],
    "Load": []
}

    df = pd.DataFrame(dict)

    return df
            

def interate_through_database(database, df):
    for i,r in database.iterrows():
        model_name = r["model_name"]
        latency = r["latency avg"]
        map = r["map"]
        api = r["api"]

        entry = {"Model": [model_name], "Average latency": [latency], "mAP": [map], "API": [api]}
        df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df







if __name__ == "__main__":
    main()
