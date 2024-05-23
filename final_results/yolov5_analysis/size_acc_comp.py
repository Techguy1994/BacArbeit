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

    fig = px.scatter(df, x="map", y="size", color="api",title="Yolov5")
    fig.show()


def create_empty_dataframe():
    dict = {
    "model_name": [],
    "size": [],
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
        #latency = r["latency avg"]
        map = r["map"]
        api = r["api"]

        model_size = get_model_size(model_name, api)

        entry = {"model_name": [model_name], "size": [model_size], "map": [map], "api": [api]}
        df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df

def get_model_size(model_name, api):
    path = os.path.abspath(__file__)
    path = path.split("final_results")[0]
    path = os.path.join(path, "models", model_name.rstrip(model_name[-1]))

    if api == "tf":
        path = os.path.join(path, "tflite")
        ending = ".tflite"
    elif api == "onnx":
        path = os.path.join(path, "onnx")
        ending = ".onnx"
    elif api == "ov":
        path = os.path.join(path, "ov")
        ending = ".xml"
    elif api == "pytorch":
        path = os.path.join(path, "pytorch")
        ending = ".pt"
    else:
        print("not found")
        sys.exit()
    
    models = os.listdir(path)

    print(models)

    for model in models:
        print(model)
        if (model_name in model) and (ending in model):
            path = os.path.join(path, model)
            print(path, model)
            model_size = os.path.getsize(path)
    
    return model_size

if __name__ == "__main__":
    main()