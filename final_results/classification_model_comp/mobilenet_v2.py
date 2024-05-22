import pandas as pd
import plotly.express as px

def main():
    database = pd.read_csv("database.csv")
    csv = "mobilenet_v2_scatter.csv"

    df = create_empty_dataframe()

    df = interate_through_database(database, df)

    print(df)
    df.to_csv(csv)

    fig = px.scatter(df, x="top1", y="mean_lat", color="model_name", symbol="api", title="MobilenNet V2")
    fig.show()

    fig = px.scatter(df, x="top5", y="mean_lat", color="api", symbol="model_name", title="MobilenNet V2")
    fig.show()


def create_empty_dataframe():
    dict = {
    "model_name": [],
    "api": [],
    "mean_lat": [],
    "top1": [],
    "top5": [],
    "quantized": []
    }

    df = pd.DataFrame(dict)

    return df

def interate_through_database(database, df):



    database = database[(database["os"] == "ubuntus")]

    

    print("----")
    print(database)
    print("---")


    models = [
        "lite-model_mobilenet_v2_100_224_fp32_1",
        "mobilenetv2-12",
        "mobilenet-v2-1.4-224_FP16",
        "mobilenet_v2",
        "lite-model_mobilenet_v2_100_224_uint8_1",
        "mobilenetv2-12-int8",
        "mobilenet_v2_q"
        ]
    
    quantized = [
        "lite-model_mobilenet_v2_100_224_uint8_1",
        "mobilenetv2-12-int8",
        "mobilenet_v2_q"
    ]

    

    for i,r in database.iterrows():
        if r["model_name"] in models:
            print(r["model_name"])
            api = r["api"]
            mean_lat = r["latency avg"]
            top1 = r["top1"]
            top5 = r["top5"]

            if r["model_name"] in quantized:
                q = "yes"
            else:
                q = "no"

            entry = {
                "model_name": [r["model_name"]],
                "api": [api],
                "mean_lat": [mean_lat],
                "top1": [top1],
                "top5": [top5],
                "quantized": [q]
            }
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 
    
    return df


if __name__ == "__main__":
    main()