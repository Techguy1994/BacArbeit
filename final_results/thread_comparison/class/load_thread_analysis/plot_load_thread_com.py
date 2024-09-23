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
    '2': 'tab:brown',
    '3': 'tab:red',
    '4': 'tab:orange'
    }

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
    boolean = booleans[0]

    if boolean == True:
        for api in apis:
            for load in loads:
                name = api + load + ".png"
                filt = (df["api"] == api) & (df["load"] == load)
                model_df = df.loc[filt]
                print(model_df.head)
                model_df.to_csv("temp_filt.csv")

                distplot = sns.displot(model_df, x="latency", hue="thread", kde=True, binwidth=0.001, stat="count")
                distplot.figure.savefig(name)
    elif boolean == "tf":
        api = apis[4]
        load = loads[0]
        name = api + load + ".png"
        filt = (df["api"] == api) & (df["load"] == load)
        model_df = df.loc[filt]
        print(model_df.head)
        model_df.to_csv("temp_filt.csv")
        model_df = model_df.sort_values(by=["thread"])
        print(model_df)
        distplot = sns.displot(model_df, x="latency", hue="thread", kind="kde", palette=palette, legend=False)
        plt.xlim(0.035,0.1)
        plt.ylim(0,1000)
        plt.ylabel("KDE")
        #plt.yscale("log")
        #plt.xscale("log")
        plt.xlabel("Latency (s)")
        plt.legend(loc="upper right", labels = label, title="Core Settings")
        plt.title("Tflite runtime")
        plt.show()
    elif boolean == "arm":
        api = apis[3]
        load = loads[0]
        name = api + load + ".png"
        filt = (df["api"] == api) & (df["load"] == load)
        model_df = df.loc[filt]
        print(model_df.head)
        model_df.to_csv("temp_filt.csv")
        model_df = model_df.sort_values(by=["thread"])
        print(model_df)
        distplot = sns.displot(model_df, x="latency", hue="thread", kind="kde", binwidth=0.0005, palette=palette, legend=False)
        plt.xlim(0.035,0.1)
        plt.ylim(0,1000)
        plt.ylabel("KDE")
        plt.xlabel("Latency (s)")
        plt.legend(loc="upper right", labels = label, title="Core Settings")
        plt.title("Armnn delegate")
        plt.show()
    elif boolean == "ov":
        api = apis[1]
        load = loads[0]
        name = api + load + ".png"
        filt = (df["api"] == api) & (df["load"] == load)
        model_df = df.loc[filt]
        print(model_df.head)
        model_df.to_csv("temp_filt.csv")
        model_df = model_df.sort_values(by=["thread"])
        print(model_df)
        distplot = sns.displot(model_df, x="latency", hue="thread", kind="kde", binwidth=0.0005, palette=palette, legend=False)
        plt.xlim(0.035,0.1)
        plt.ylim(0,1000)
        plt.ylabel("KDE")
        plt.xlabel("Latency (s)")
        plt.legend(loc="upper right", labels = label, title="Core Settings")
        plt.title("Openvino")
        plt.show()
    elif boolean == "onnx":
        api = apis[0]
        load = loads[0]
        name = api + load + ".png"
        filt = (df["api"] == api) & (df["load"] == load)
        model_df = df.loc[filt]
        print(model_df.head)
        model_df.to_csv("temp_filt.csv")
        model_df = model_df.sort_values(by=["thread"])
        print(model_df)
        distplot = sns.displot(model_df, x="latency", hue="thread", kind="kde", binwidth=0.0005, palette=palette, legend=False)
        plt.xlim(0.035,0.1)
        plt.ylim(0,1000)
        plt.ylabel("KDE")
        plt.xlabel("Latency (s)")
        plt.legend(loc="upper right", labels = label, title="Core Settings")
        plt.title("Onnx runtime")
        plt.show()
    elif boolean == "torch":
        print("heyyyo")
        api = apis[2]
        load = loads[0]
        name = api + load + ".png"
        print(api)
        filt = (df["api"] == api) & (df["load"] == load)
        model_df = df.loc[filt]
        print(model_df.head)
        model_df.to_csv("temp_filt.csv")
        model_df = model_df.sort_values(by=["thread"])
        print(model_df)
        distplot = sns.displot(model_df, x="latency", hue="thread", kind="kde", binwidth=0.0005, palette=palette, legend=False)
        #plt.xlim(0.03,0.1)
        plt.ylim(0,1000)
        plt.ylabel("KDE")
        plt.xlabel("Latency (s)")
        plt.legend(loc="upper right", labels = label, title="Core Settings")
        plt.title("PyTorch")
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
