import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd 
import sys
import os
import numpy as np


def main():
    database_delegate = pd.read_csv("database_delegate.csv")
    database_tf = pd.read_csv("database_tf.csv")

    database = pd.concat([database_delegate, database_tf], axis=0)

    

    df = create_empty_dataframe()

   

    df = interate_through_database(database, df)
    df.to_csv("temp.csv")

    model_name = "lite_model_mobilenet_v3_large_100_224_fp32_1"
    filt = (df["model_name"] == model_name)
    model_df = df.loc[filt]

    model_df.to_csv("model_temp.csv")




 



def create_empty_dataframe():
    dict = {
    "model_name": [],
    "latency": [],
    "thread": []
}

    df = pd.DataFrame(dict)

    return df

def filter_through_database(df):
    #not used
    list_of_dfs = []

    model_names = df.model_name.unique()
    threads = df.thread.unique()

    print(model_names)
    print(threads)

    for model_name in model_names:
        #for thread in threads:
        #    filt = (df["model_name"] == model_name) & (df["thread"] == thread)

        model_df =  df["model_name"] == model_name
        list_of_dfs.append(0)
            



def interate_through_database(database, df):
    for i,r in database.iterrows():
        model_name = r["model_name"]
        latency_link = r["latency"]
        threads = r["thread count"]

        lat = pd.read_csv(latency_link)

        for ii, rr in lat.iterrows():
            inference_time = rr["inference time"]

            entry = {"model_name": [model_name], "thread": [threads], "latency": [inference_time]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df

def get_values_for_plot(df):
    model_names = df.model_name.unique()
    threads = df.thread.unique()
    df_names = []
    #dfs = []

    for model_name in model_names:
        for thread in threads:
            
            filt = (df["model_name"] == model_name) & (df["thread"] == thread)
            
            name = model_name + "_T" + str(int(thread))

            
            name = name.replace("-", "_")
            globals()[name] = df.loc[filt]
        
            df_names.append(name)


    return df_names




if __name__ == "__main__":
    main()
