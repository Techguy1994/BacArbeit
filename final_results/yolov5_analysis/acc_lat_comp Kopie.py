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

    bool = True

    if bool:
        fig = px.line(df, x="mAP", y="Average latency", color="API", markers=True)
        fig.update_layout(
        xaxis=dict(range=[0.15, 0.5]),
        yaxis=dict(range=[0, 7])  # Set the x-axis range from 2 to 8
        )
        fig.show()
    else:
        fig = px.bar(df, x="Model", y="Average latency", color="API", barmode="group")
        fig.show()




def create_empty_dataframe():
    dict = {
    "Model": [],
    "Average latency": [],
    "API": [],
    "mAP": []
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

        entry = {"Model": [model_name], "Average latency": [latency], "mAP": [map], "API": [api]}
        df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df

if __name__ == "__main__":
    main()