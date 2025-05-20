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

    csv_name = "temp.csv"

    if os.path.isfile(csv_name):
        df = pd.read_csv(csv_name)
    else:        
        df = create_empty_dataframe()
        df = interate_through_database(database, df)
        df.to_csv("temp.csv")

    fws = df.Frameworks.unique()
    #threads = df.thread.unique()

    apis = ["ONNX", "OpenVINO", "PyTorch", "Arm NN Delegate", "TFLite"]
    loads = ["default", "one", "two", "three"]
    core_order = ["Default", "Four Cores", "Three Cores", "Two Cores", "One Core"]
    palette = {
    "Default": "#1f77b4",       # blue
    "Four Cores": "#ff7f0e",    # orange
    "Three Cores": "#2ca02c",   # green
    "Two Cores": "#d62728",     # red
    "One Core": "#9467bd"       # purple
    }

    booleans = ["TFLite", "Arm NN Delegate", "PyTorch", "OpenVINO", "ONNX", True]
    boolean = booleans[1]

    if boolean == True:
        for api in apis:
            for load in loads:
                name = api + load + ".pdf"
                filt = (df["api"] == api) & (df["load"] == load)
                model_df = df.loc[filt]
                print(model_df.head)
                model_df.to_csv("temp_filt.csv")

                distplot = sns.displot(model_df, x="latency", hue="thread", kde=True, binwidth=0.001, stat="count")
                plt.ylabel("Load on number of cores", fontsize=20)
                plt.xlabel("Latency (s)", fontsize=20)
                distplot.figure.savefig(name)
    else:
        if boolean == "TFLite":
            api = apis[4]
        elif boolean == "Arm NN Delegate":
            api = apis[3]
        elif boolean == "OpenVINO":
            api = apis[1]
        elif boolean == "ONNX":
            api = apis[0]
        elif boolean == "PyTorch":
            api = apis[2]
            plt.xlim(0.1,0.3)
            plt.ylim(0,10)
            plt.legend(loc="upper right", labels = core_order, title="Number of cores set in code")


    load = "No Load"
    name = api + load + ".pdf"
    filt = (df["Frameworks"] == api) & (df["Load"] == load)
    model_df = df.loc[filt]

    # Convert thread column to ordered categorical
    model_df["thread"] = pd.Categorical(model_df["thread"], categories=core_order, ordered=True)

# Now sort the DataFrame by this order
    model_df = model_df.sort_values(by="thread")

    model_df.to_csv("temp_filt.csv")

    # Plot and get the FacetGrid object
    distplot = sns.displot(
        model_df, 
        x="latency", 
        hue="thread", 
        kind="kde", 
        hue_order=core_order,
        palette=palette, 
        linewidth=5,
        height=6, 
        aspect=1.5,
        legend=True
    )

    # Set limits and labels using distplot.ax
    distplot.ax.set_xlim(0.035, 0.1)
    distplot.ax.set_ylim(0, 500)
    distplot.ax.set_ylabel("KDE (kernel density estimation)", fontsize=20)
    distplot.ax.set_xlabel("Latency [s]", fontsize=20)
    distplot.ax.tick_params(axis='x', labelsize=14)
    distplot.ax.tick_params(axis='y', labelsize=14)

    # Set title
    if boolean == "TFLite":
        distplot.ax.set_title("MobileNetv3 Large - TFLite", fontsize=24)
    elif boolean == "Arm NN Delegate":
        distplot.ax.set_title("MobileNetv3 Large - Armnn NN Delegate", fontsize=24)
    elif boolean == "OpenVINO":
        distplot.ax.set_title("MobileNetv3 Large - OpenVINO", fontsize=24)
    elif boolean == "ONNX":
        distplot.ax.set_title("MobileNetv3 Large - ONNX", fontsize=24)
        distplot.ax.legend(title="Number of cores for inference", title_fontsize=24, fontsize=20, loc="upper left", labels=core_order)
    elif boolean == "PyTorch":
        distplot.ax.set_xlim(0.1, 0.3)
        distplot.ax.set_ylim(0, 10)
        distplot.ax.set_title("MobileNetv3 Large - PyTorch", fontsize=24)
        distplot.ax.legend(title="Number of cores for inference", fontsize=20, labels=core_order)

    # Automatically get handles and labels from the plot
    handles, labels = distplot.ax.get_legend_handles_labels()

    print(handles, labels)

    # Sort them by the order in core_order to ensure consistent order
    label_order = [label for label in core_order if label in labels]
    handles_ordered = [handles[labels.index(label)] for label in label_order]

    # Add the legend with correct labels and matching colors
    distplot.ax.legend(
        handles=handles_ordered,
        labels=label_order,
        title="Number of cores for inference",
        title_fontsize=16,
        fontsize=14,
        loc='upper right',
        borderaxespad=0.
    )


    """
    # Add legend if not already set above
    if boolean not in ["ONNX", "PyTorch"]:
        distplot.ax.legend(
        title="Number of cores for inference",
        title_fontsize=16,
        fontsize=14,
        labels=core_order,
        #bbox_to_anchor=(1.05, 1),  # position legend outside
        loc='upper right',
        borderaxespad=0.
    )
    """

    # Save
    distplot.savefig(name, bbox_inches='tight')


def create_empty_dataframe():
    dict = {
    "Frameworks": [],
    "latency": [],
    "thread": [],
    "Load": []
}

    df = pd.DataFrame(dict)

    return df
            



def interate_through_database(database, df):
    for i,r in database.iterrows():
        fw = r["Frameworks"]
        latency_link = r["latency"]
        threads = r["Core Count"]
        l = r["Load"]

        print(fw, threads, latency_link)

        lat = pd.read_csv(latency_link)

        for ii, rr in lat.iterrows():
            inference_time = rr["inference time"]

            entry = {"Frameworks": [fw],"latency": [inference_time], "thread": [threads], "Load": [l]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df



if __name__ == "__main__":
    main()
