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

    palette = {
    'False': 'tab:blue',
    '1': 'tab:green',
    '2': 'tab:purple',
    '3': 'tab:red',
    '4': 'tab:orange'
    }

    csv_name = "temp.csv"

    if os.path.isfile(csv_name):
        df = pd.read_csv(csv_name)
    else:        
        df = create_empty_dataframe()
        df = interate_through_database(database, df)
        df.to_csv("temp.csv")

    apis = df.api.unique()
    threads = df.thread.unique()
    print(apis)
    print(threads)
    print(df.head)

    apis = ["onnx", "ov", "pytorch", "delegate", "tf"]
    loads = ["default", "one", "two", "three"]
    label = ["Default", "4 cores", "3 cores", "2 cores", "1 core"]

    booleans = ["tf", "arm", "torch", "ov", "onnx", True]
    boolean = booleans[5]

    if boolean == True:
        for api in apis:
            for load in loads:
                name = api + load + ".png"
                filt = (df["api"] == api) & (df["load"] == load)
                model_df = df.loc[filt]
                print(model_df.head)
                model_df.to_csv("temp_filt.csv")

                distplot = sns.displot(model_df, x="latency", hue="thread", kde=True, binwidth=0.001, stat="count")
                plt.ylabel("Load on number of cores", fontsize=20)
                plt.xlabel("Latency (s)", fontsize=20)
                distplot.figure.savefig(name)
    else:
        if boolean == "tf":
            api = apis[4]
        elif boolean == "arm":
            api = apis[3]
        elif boolean == "ov":
            api = apis[1]
        elif boolean == "onnx":
            api = apis[0]
        elif boolean == "torch":
            api = apis[2]
            plt.xlim(0.1,0.3)
            plt.ylim(0,10)
            plt.legend(loc="upper right", labels = label, title="Core Settings set in Code")


        load = loads[0]
        name = api + load + ".png"
        filt = (df["api"] == api) & (df["load"] == load)
        model_df = df.loc[filt]
        model_df.to_csv("temp_filt.csv")
        model_df = model_df.sort_values(by=["thread"])
        distplot = sns.displot(model_df, x="latency", hue="thread", kind="kde", palette=palette, legend=False, linewidth=5)
        plt.xlim(0.035,0.1)
        plt.ylim(0,500)
        plt.ylabel("KDE (kernel density estimation)", fontsize=20)
        plt.xlabel("Latency (s)", fontsize=20)
        plt.legend(loc="upper right", labels = label, title="Core Settings", title_fontsize=24, fontsize=20)
        plt.xticks(fontsize=14)  # Increase x-axis tick labels
        plt.yticks(fontsize=14) 
        plt.tight_layout()

        if boolean == "tf":
            plt.title("Tflite runtime", fontsize=24)
        elif boolean == "arm":
            plt.title("Armnn delegate", fontsize=24)
        elif boolean == "ov":
            plt.title("Openvino", fontsize=24)
        elif boolean == "onnx":
            plt.title("Onnx runtime", fontsize=24)
            plt.legend(loc="upper left", labels = label, title="Core Settings", title_fontsize=24, fontsize=20)
        elif boolean == "torch":
            plt.xlim(0.1,0.3)
            plt.ylim(0,10)
            plt.legend(loc="upper right", labels = label, title="Core Settings set in Code")
            plt.title("PyTorch", fontsize=24)
        
        plt.show()


def create_empty_dataframe():
    dict = {
    "model_name": [],
    "latency": [],
    "thread": [],
    "api": [],
    "load": []
}

    df = pd.DataFrame(dict)

    return df
            



def interate_through_database(database, df):
    for i,r in database.iterrows():
        model_name = r["model_name"]
        latency_link = r["latency"]
        threads = r["thread count"]
        api = r["api"]
        l = r["load"]

        print(model_name, threads, api, latency_link)

        lat = pd.read_csv(latency_link)

        for ii, rr in lat.iterrows():
            inference_time = rr["inference time"]

            entry = {"model_name": [model_name],"latency": [inference_time], "thread": [threads], "api": [api], "load": [l]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df



if __name__ == "__main__":
    main()
