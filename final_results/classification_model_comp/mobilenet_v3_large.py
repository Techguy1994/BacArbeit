import pandas as pd
import plotly.express as px

def main():
    database = pd.read_csv("database.csv")
    csv = "mobilenet_v3_large_scatter.csv"

    df = create_empty_dataframe()

    df = interate_through_database(database, df)

    print(df)
    df.to_csv(csv)

    fig = px.scatter(df, x="top1", y="mean_lat", color="api", symbol="quantized", title="MobilenNet V3 Large")
    fig.update_traces(marker=dict(size=20),
                  selector=dict(mode='markers'))
    fig.show()

    fig = px.scatter(df, x="top5", y="mean_lat", color="api", symbol="quantized", title="MobilenNet V3 Large")
    fig.update_traces(marker=dict(size=20),
                  selector=dict(mode='markers'))
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
        "lite-model_mobilenet_v3_large_100_224_fp32_1",
        "mobilenetv3_large_100_Opset17",
        "mobilenet-v3-large-1.0-224-tf_FP32",
        "mobilenet_v3_large",
        "lite-model_mobilenet_v3_large_100_224_uint8_1",
        "mobilenet_v3_large_q"
        ]
    
    quantized = [
        "lite-model_mobilenet_v3_large_100_224_uint8_1",
        "mobilenet_v3_large_q"
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
                "model_name": ["mobilenet_v3_large"],
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