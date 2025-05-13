import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd 
import sys
import os
import numpy as np


def main():
    database = pd.read_csv("thread_comparison_database.csv")

    

    df = create_empty_dataframe()

   

    df = interate_through_database(database, df)
    df.to_csv("temp.csv")

    model_name = "mobilenet_v3_large"
    #deprecated
    values = []

    #model_name = ["lite-model_mobilenet_v3_large_100_224_fp32_1"]
    threads = ["3", "4"]
    df_names = get_values_for_plot(df)

    #print(df_names)

    for df_name in df_names:
        for thread in threads:
            temp = model_name + "_T" + thread
            if temp in df_name:
                #print(eval(temp))
                #print(eval(temp).latency.values.tolist())
                values.append(eval(temp).latency.values.tolist())

    #print(len(values[0]))
                              
    fig = ff.create_distplot(values, threads, bin_size=0.00001, curve_type="normal")
    print(fig)
    fig.show()


    #trying with px 
    #filter_through_database(df)

    filt = (df["model_name"] == model_name)
    model_df = df.loc[filt]

    print(model_df.head)

    fig = px.histogram(model_df, x="latency", color="thread", marginal="box", barmode="overlay", nbins=5000, title=model_name)
    fig.update_traces(opacity=0.75)
    fig.show()



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
