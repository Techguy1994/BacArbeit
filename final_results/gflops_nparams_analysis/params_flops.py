import pandas as pd
import plotly.express as px
import sys
import os

def main():
    database = pd.read_csv("database.csv")
    csv = "test.csv"

    df = create_empty_dataframe()

    df = interate_through_database(database, df)

    print(df)
    df.to_csv(csv)

    fig = px.scatter(df, x="GFlops", y="NParams in Million", size="mean_lat", color="model_name", hover_name="model_name", size_max=60)
    fig.show()
    fig = px.scatter(df, x="GFlops", y="NParams in Million", size="map", color="model_name", hover_name="model_name", size_max=60)
    fig.show()


def create_empty_dataframe():
    dict = {
    "model_name": [],
    "mean_lat": [],
    "map": [],
    "GFlops": [],
    "NParams in Million": []
    }

    df = pd.DataFrame(dict)

    return df



def interate_through_database(database, df):

    database = database[(database["api"] == "ov") & (database["inference type"] == "det")]

    for i,r in database.iterrows():
        #print(r["model_name"])

        gflops_mparams_csv = "GFLOPS_MPARAMS.csv"
        gflops_mparams_df = pd.read_csv(gflops_mparams_csv, delimiter=";")

        gflops = gflops_mparams_df[gflops_mparams_df["model"] == r["model_name"]]["GFLOPS"].item()
        mparams = gflops_mparams_df[gflops_mparams_df["model"] == r["model_name"]]["MParams"].item()

        entry = {
                "model_name": r["model_name"],
                "mean_lat": [r["latency avg"]],
                "map": [r["map"]],
                "GFlops": [gflops],
                "NParams in Million": [mparams]
            }
        
        df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 
    
    return df


if __name__ == "__main__":
    main()