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

            filtered_thread_default_db = filtered_db[filtered_db["thread count"] == "False"]
            for i,r in filtered_thread_default_db.iterrows():
                avg = r["latency avg"]
                threads = r["thread count"]

            entry = {"mean latency": [avg], "thread": [threads], "api": [api], "load": [load]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True)





    df.to_csv("tabel_two.csv")

    
            



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
