import pandas as pd
import sys

def main():
    database = pd.read_csv("database_os_comparison.csv")
    csv = "temp_normed_comparison.csv"

    df = create_empty_dataframe()

    df = interate_through_database(database, df)
    df.to_csv(csv)

    calculate_normed_for_each_model(df)
    #calculate_normed_for_os(df)
    #calculate_normed_for_each_framework(df)

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

def calculate_normed_for_each_model_old(df):
    os = df.os.unique()
    models = df.model_name.unique()
    #n = df.shape[0]
    #print(n)
    os_dfs = []
    #print(os)
    #print(models)
    median_lists = []
    normed = []

    os_dfs.append(df.loc[df["os"] == "ubuntus"])
    os_dfs.append(df.loc[df["os"] == "raspberryos"])
    n = os_dfs[0].shape[0]

    os_dfs[0].to_csv("ubuntu.csv")
    os_dfs[1].to_csv("raspberryos.csv")

    for os_df in os_dfs:
        normed_list = []
        for model in models:
            #print(model)
            median_model = os_df.loc[os_df["model_name"] == model]
            median_model.to_csv("test.csv")
            median = median_model["latency"].median()
            temp = median_model["latency"].div(median)
            #print(temp)
            temp = temp.sum()
            normed_list.append(temp)
          
        #print(normed_list)
        sum_normed_list = sum(normed_list)
        normed.append(sum_normed_list/n)
    
    print(normed)

def calculate_normed_for_each_model(df):
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
        #print(temp)
    
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



def calculate_normed_for_each_framework(df):
    os_dfs = []
    medians = []
    normed = []
    

    os_dfs.append(df.loc[df["os"] == "ubuntus"])
    os_dfs.append(df.loc[df["os"] == "raspberryos"])
    n = os_dfs[0].shape[0]

    apis = df.api.unique()
    print(apis)

    for api in apis:
        temp = os_dfs[0].loc[os_dfs[0]["api"] == api]["latency"].median()
        medians.append(temp)
        print(temp)
        
    
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

    

def calculate_normed_for_os(df):
    
    os_dfs = []
    normed = []

    os_dfs.append(df.loc[df["os"] == "ubuntus"])
    os_dfs.append(df.loc[df["os"] == "raspberryos"])
    n = os_dfs[0].shape[0]
    print(n)

    ubuntu_median = os_dfs[0]["latency"].median()
    print("median", ubuntu_median)

    for os_df in os_dfs:
        temp = os_df["latency"].div(ubuntu_median)
        temp = temp.sum()
        print(temp)
        temp = temp/n
        print(temp)
        normed.append(temp)
    
    print(normed)



if __name__ == "__main__":
    main()