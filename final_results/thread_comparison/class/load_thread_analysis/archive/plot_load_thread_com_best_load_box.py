import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd 
import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():

    if os.path.exists("f.csv"):
        df = pd.read_csv("f.csv")
    else:
        database = pd.read_csv("load_database.csv")



        db = database[database['api'] != "pytorch"]
        db = db.drop('os', axis=1)
        db = db.drop('inference type', axis=1)
        db = db.drop('top1', axis=1)
        db = db.drop('top5', axis=1)
        db = db.drop('miou', axis=1)
        db = db.drop('map', axis=1)
        db = db.drop('api', axis=1)

        load_mapping = {
        'default': 'No load',
        'one': '1 thread load',
        'two': '2 thread load',
        'three': '3 thread load'
        }

        # Apply the mapping
        db['load'] = db['load'].replace(load_mapping)
        db['thread count'] = db['thread count'].replace({"False": 'Default'})

        db.to_csv("t.csv")

        df = create_empty_dataframe()
        df = interate_through_database(db, df)

        df.to_csv("f.csv")



    """
    unique_apis = db['api'].unique()
    unique_loads = db["load"].unique()

    sort_loads = ["default", "one", "two", "three"]

    db = db.sort_values('load')

    load_order = ['default', 'one', 'two', 'three']
    db['load'] = pd.Categorical(db['load'], categories=load_order, ordered=True)

    # 2. Fix the 'thread count' order
    thread_order = ['False', '1', '2', '3', '4']
    db['thread count'] = pd.Categorical(db['thread count'], categories=thread_order, ordered=True)
    print(unique_apis)
    """

    thread_order = ['Default', '1', '2', '3', '4']
    df['thread count'] = pd.Categorical(df['thread count'], categories=thread_order, ordered=True)

    load_order = ['No load', '1 thread load', '2 thread load', '3 thread load']
    df['Load'] = pd.Categorical(df['Load'], categories=load_order, ordered=True)


    
    
    g = sns.catplot(
        data=df, kind="box",
        x="Load", y="latency", hue="thread count",
        col="API", col_wrap=2,  # 2x2 layout since we have 4 frameworks now
        height=4, aspect=1.2, palette="viridis"
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Artificial Load", "Latency (ms)")
    g._legend.set_title("Thread Count")
    plt.tight_layout()
    plt.show()


        


def create_empty_dataframe():
    dict = {
    "latency": [],
    "thread count": [],
    "API": [],
    "Load": []
}

    df = pd.DataFrame(dict)

    return df

def interate_through_database(database, df):
    for i,r in database.iterrows():

        load = r["load"]
        thread = r["thread count"]
        api = r["model_name"]
        latency_link = r["latency"]

        lat = pd.read_csv(latency_link)

        for ii, rr in lat.iterrows():
            inference_time = rr["inference time"]

            entry = {"API": [api], "thread count": thread, "Load": load, "latency": [inference_time]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df
            







if __name__ == "__main__":
    main()
