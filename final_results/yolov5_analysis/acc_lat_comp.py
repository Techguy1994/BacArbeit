import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd 
import sys
import os
import numpy as np

def main():
    database = pd.read_csv("database.csv")
    df = create_empty_dataframe()

    database = filter_through_database(database)
    database = database.sort_values("latency avg")
    df = interate_through_database(database, df)
    df.to_csv("temp.csv")

    fig = px.line(df, x="map", y="latency", color="api",title="Yolov5")
    fig.show()


def create_empty_dataframe():
    dict = {
    "model_name": [],
    "latency": [],
    "api": [],
    "map": []
}

    df = pd.DataFrame(dict)

    return df

def filter_through_database(database):
    database = database.loc[database['inference type'] == "det"]
    print(database)
    return database

def interate_through_database(database, df):
    for i,r in database.iterrows():
        model_name = r["model_name"]
        latency = r["latency avg"]
        map = r["map"]
        api = r["api"]

        entry = {"model_name": [model_name], "latency": [latency], "map": [map], "api": [api]}
        df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df

if __name__ == "__main__":
    main()