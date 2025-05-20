import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd 
import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap


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
    boolean = booleans[4]

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


    load = "No Load"
    name = api + load + ".pdf"
    filt = (df["Frameworks"] == api) & (df["Load"] == load)
    model_df = df.loc[filt]

    # Convert thread column to ordered categorical
    model_df["thread"] = pd.Categorical(model_df["thread"], categories=core_order, ordered=True)

# Now sort the DataFrame by this order
    model_df = model_df.sort_values(by="thread")

    model_df.to_csv("temp_filt.csv")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot KDEs manually using sns.kdeplot
    for thread in core_order:
        thread_df = model_df[model_df["thread"] == thread]
        if not thread_df.empty:
            sns.kdeplot(
                data=thread_df,
                x="latency",
                ax=ax,
                linewidth=3,
                label=thread,
                color=palette[thread],
                fill=True
            )

    # Set plot labels and limits
    ax.set_xlim(0.035, 0.1)
    ax.set_ylim(0, 2500)
    ax.set_ylabel("KDE (kernel density estimation)", fontsize=20)
    ax.set_xlabel("Latency [s]", fontsize=20)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Title
    ax.set_title("MobileNetV3 Large - " + api, fontsize=24)

    # Add the correct legend
    if boolean not in ["ONNX", "PyTorch"]:
        ax.legend(
            title="Number of cores for inference",
            title_fontsize=16,
            fontsize=14,
            loc="upper right"
        )
    elif boolean == "ONNX":
        ax.legend(
            title="Number of cores for inference",
            title_fontsize=16,
            fontsize=14,
            loc="upper left"
        )
    elif boolean == "PyTorch":
        ax.set_xlim(0.1, 0.3)
        ax.set_ylim(0, 60)
        ax.legend(
            title="Number of cores for inference",
            title_fontsize=16,
            fontsize=14,
            loc="upper right"
        )

        long_text = "Note: PyTorch does not behave as expected, all cores are loaded regardless which core count is given"
        wrapped = "\n".join(textwrap.wrap(long_text, width=50))
        ax.text(
            0.01, 0.98,  # X and Y in axis coordinates (0 to 1)
            wrapped,
            fontsize=14,
            color="red",
            ha='left',
            va='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='red')
        )

    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    print(name)

    # Save the figure
    fig.tight_layout()
    fig.savefig(name, bbox_inches='tight')


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
