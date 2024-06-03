import pandas as pd
import sys
import numpy as np
import plotly.graph_objects as go

def main():
    database = pd.read_csv("database_os_comparison.csv")
    csv = "temp_normed_comparison.csv"

    df = create_empty_dataframe()

    df = interate_through_database(database, df)
    df.to_csv(csv)

    #calculate_normed_for_each_model_mean(df)
    fw_means, normed, apis = calculate_normed_for_each_framework_mean(df)

    fw_means[0].update({"ubuntu": normed[0]})
    fw_means[1].update({"raspberryos": normed[1]})

    print(apis)
    apis = apis.tolist()
    apis = apis.append("all")

    fw_means_list = fw_means[1].values()
    print(fw_means)

    """
    fig = go.Figure(data=[go.Table(header=dict(values=['', 'T_raspberry_os']),
                 cells=dict(values=[apis, fw_means_list]))
                     ])
    
    fig.show()
    """
 

def create_empty_dataframe():
    dict = {
    "model_name": [],
    "os": [],
    "api": [],
    "latency": []
}

    df = pd.DataFrame(dict)

    return df

def interate_through_database(database, df):
    for i,r in database.iterrows():
        model_name = r["model_name"]
        os = r["os"]
        api = r["api"]
        latency_link = r["latency"]

        lat = pd.read_csv(latency_link)

        for ii, rr in lat.iterrows():
            inference_time = rr["inference time"]

            entry = {"model_name": [model_name], "os": [os], "api": [api], "latency": [inference_time]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df

def calculate_normed_for_each_model_median(df):
    os_dfs = []
    medians = []
    normed = []
    

    os_dfs.append(df.loc[df["os"] == "ubuntus"])
    os_dfs.append(df.loc[df["os"] == "raspberryos"])
    n = os_dfs[0].shape[0]

    models = df.model_name.unique()

    for model in models:
        temp = os_dfs[0].loc[os_dfs[0]["model_name"] == model]["latency"].median()
        medians.append(temp)
    
    for os_df in os_dfs:
        i = 0
        temp_list = []
        
        for model in models:
            temp = os_df.loc[os_df["model_name"] == model]
            temp = temp["latency"].div(medians[i])
            temp = temp.sum()
            temp_list.append(temp)

            i = i + 1
        
        temp = sum(temp_list)/n
        normed.append(temp)

    
    print(normed)



def calculate_normed_for_each_framework_median(df):
    os_dfs = []
    medians = []
    normed = []
    

    os_dfs.append(df.loc[df["os"] == "ubuntus"])
    os_dfs.append(df.loc[df["os"] == "raspberryos"])
    n = os_dfs[0].shape[0]

    apis = df.api.unique()

    for api in apis:
        temp = os_dfs[0].loc[os_dfs[0]["api"] == api]["latency"].median()
        medians.append(temp)
    
    for os_df in os_dfs:
        i = 0
        temp_list = []
        
        for api in apis:
            temp = os_df.loc[os_df["api"] == api]
            temp = temp["latency"].div(medians[i])
            temp = temp.sum()
            temp_list.append(temp)

            i = i + 1
        
        temp = sum(temp_list)/n
        normed.append(temp)

    
    print(normed)

    

def calculate_normed_for_os_median(df):
    
    os_dfs = []
    normed = []

    os_dfs.append(df.loc[df["os"] == "ubuntus"])
    os_dfs.append(df.loc[df["os"] == "raspberryos"])
    n = os_dfs[0].shape[0]

    ubuntu_median = os_dfs[0]["latency"].median()

    for os_df in os_dfs:
        temp = os_df["latency"].div(ubuntu_median)
        temp = temp.sum()
        temp = temp/n
        normed.append(temp)
    
    print(normed)

def calculate_normed_for_each_model_mean(df):
    os_dfs = []
    means = []
    normed = []
    

    os_dfs.append(df.loc[df["os"] == "ubuntus"])
    os_dfs.append(df.loc[df["os"] == "raspberryos"])
    n = os_dfs[0].shape[0]

    models = df.model_name.unique()

    for model in models:
        temp = os_dfs[0].loc[os_dfs[0]["model_name"] == model]["latency"].mean()
        means.append(temp)
    
    for os_df in os_dfs:
        i = 0
        temp_list = []
        
        for model in models:
            temp = os_df.loc[os_df["model_name"] == model]
            temp = temp["latency"].div(means[i])
            temp = temp.sum()
            temp_list.append(temp)

            i = i + 1
        
        temp = sum(temp_list)/n
        temp = temp.round(3)
        normed.append(temp)

    
    print(normed)



def calculate_normed_for_each_framework_mean(df):
    os_dfs = []
    means = []
    fw_means = []
    normed = []
    

    os_dfs.append(df.loc[df["os"] == "ubuntus"])
    os_dfs.append(df.loc[df["os"] == "raspberryos"])
    n = os_dfs[0].shape[0]

    apis = df.api.unique()

    for api in apis:
        temp = os_dfs[0].loc[os_dfs[0]["api"] == api]["latency"].mean()
        means.append(temp)
    
    for os_df in os_dfs:
        i = 0
        temp_list = []
        temp_fw_list = {}
        
        for api in apis:
            temp = os_df.loc[os_df["api"] == api]
            n_api = temp.shape[0]
            temp = temp["latency"].div(means[i])
            temp = temp.sum()
            temp_list.append(temp)
            temp_fw_list.update({api: (temp/n_api).round(3)})

            i = i + 1
        
        temp = sum(temp_list)/n
        temp = temp.round(3)
        normed.append(temp)
        fw_means.append(temp_fw_list)

    print(fw_means)
    print(normed)

    return fw_means, normed, apis

    

def calculate_normed_for_os_mean(df):
    
    os_dfs = []
    normed = []

    os_dfs.append(df.loc[df["os"] == "ubuntus"])
    os_dfs.append(df.loc[df["os"] == "raspberryos"])
    n = os_dfs[0].shape[0]

    ubuntu_mean = os_dfs[0]["latency"].mean()

    for os_df in os_dfs:
        temp = os_df["latency"].div(ubuntu_mean)
        temp = temp.sum()
        temp = temp/n
        temp = temp.round(3)
        normed.append(temp)
    
    print(normed)



if __name__ == "__main__":
    main()