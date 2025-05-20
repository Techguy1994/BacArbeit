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
