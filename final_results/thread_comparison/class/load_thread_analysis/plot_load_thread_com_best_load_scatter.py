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




    print(unique_apis)

    for api in unique_apis:
        for load in unique_loads:
            filtered_db = db[(db["api"] == api) & (db["load"] == load)]
            max_value = filtered_db["latency avg"].min()
            print(max_value, api, load)

            filtered_max_db = filtered_db[filtered_db["latency avg"] == max_value]

            for i,r in filtered_max_db.iterrows():
                threads = r["thread count"]
           
            entry = {"mean latency": [max_value], "thread": [threads], "api": [api], "load": [load]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True)
    
    df['load'] = pd.Categorical(df['load'], categories=sort_loads, ordered=True)
    df = df.sort_values('load')

    df.to_csv("scatter.csv")

    # start of the scatter plot

    fig = px.scatter(df, x="load", y='mean latency', color='api', symbol="thread", title='Scatter Plot Example')
    for api in df['api'].unique():
        color_df = df[df['api'] == api]
        fig.add_scatter(x=color_df['load'], y=color_df['mean latency'], mode='lines', name=f'Line for {api}')

    fig.update_traces(marker=dict(size=15))
    # Show the plot
    fig.show()

            



    """
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
    """


        


def create_empty_dataframe():
    dict = {
    "mean latency": [],
    "thread": [],
    "api": [],
    "load": []
}

    df = pd.DataFrame(dict)

    return df
            



def interate_through_database(database, df):
    for i,r in database.iterrows():
        avg = r["latency avg"]
        latency_link = r["latency"]
        threads = r["thread count"]
        api = r["api"]
        l = r["load"]

        print(avg, threads, api)

        lat = pd.read_csv(latency_link)

        for ii, rr in lat.iterrows():
            inference_time = rr["inference time"]

            entry = {"mean latency": [avg],"latency": [inference_time], "thread": [threads], "api": [api], "load": [l]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df



if __name__ == "__main__":
    main()
