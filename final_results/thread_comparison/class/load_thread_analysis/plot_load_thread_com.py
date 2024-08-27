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

    df = interate_through_database(database, df)
    df.to_csv("temp.csv")

    apis = df.api.unique()
    threads = df.thread.unique()
    print(apis)
    print(threads)
    print(df.head)

    apis = ["onnx", "ov", "pytroch", "delegate", "tf"]
    loads = ["default", "one", "two", "three"]

    boolean = False

    if boolean == True:
        for api in apis:
            for load in loads:
                name = api + load + ".png"
                filt = (df["api"] == api) & (df["load"] == load)
                model_df = df.loc[filt]
                print(model_df.head)
                model_df.to_csv("temp_filt.csv")

                distplot = sns.displot(model_df, x="latency", hue="thread", kde=True, binwidth=0.0001, stat="count")
                distplot.figure.savefig(name)
    else:
        api = apis[1]
        load = loads[1]
        name = api + load + ".png"
        filt = (df["api"] == api) & (df["load"] == load)
        model_df = df.loc[filt]
        print(model_df.head)
        model_df.to_csv("temp_filt.csv")
        distplot = sns.displot(model_df, x="latency", hue="thread", kde=True, binwidth=0.0001, stat="count")
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

        print(model_name, threads, api)

        lat = pd.read_csv(latency_link)

        for ii, rr in lat.iterrows():
            inference_time = rr["inference time"]

            entry = {"model_name": [model_name],"latency": [inference_time], "thread": [threads], "api": [api], "load": [l]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df



if __name__ == "__main__":
    main()
