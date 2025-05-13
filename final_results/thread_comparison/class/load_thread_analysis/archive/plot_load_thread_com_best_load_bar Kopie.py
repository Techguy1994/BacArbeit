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

    df.to_csv("bar.csv")

    # Create the bar plot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="load", y="mean latency", hue="api", dodge=True, ci=None)

    # Adjust labels and title
    plt.xlabel("Load")
    plt.ylabel("Mean Latency")
    plt.title("Mean Latency by Load and API")
    plt.legend(title="API", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # start of the scatter plot


        


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
