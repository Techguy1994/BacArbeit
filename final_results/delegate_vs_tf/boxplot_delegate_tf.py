import plotly.express as px
import pandas as pd 
import sys
import os


def main():
    database = pd.read_csv("database_os_comparison.csv")
    print(database)

    df = create_empty_dataframe()
    print(df)

    db = filter_through_database(df)

    df = interate_through_database(database, df)

    print(df)
    df.to_csv("temp.csv")

    fig = px.box(df, x="model_name", y="latency", color="os", range_y=[0.005,0.1])
    fig.show()

    



    #df = px.data.tips()

    #print(df)
    #print(df.columns)
    #print(df.time.unique())

    #fig = px.box(df, y="total_bill")
    #fig.show()

    #fig = px.box(df, x="time", y="total_bill")
    #fig.show()

    #fig = px.box(df, x="day", y="total_bill", color="smoker")
    #fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
    #fig.show()

def create_empty_dataframe():
    dict = {
    "model_name": [],
    "api": [],
    "latency": []
}

    df = pd.DataFrame(dict)

    return df

def filter_through_database(df):
    pass

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




if __name__ == "__main__":
    main()
