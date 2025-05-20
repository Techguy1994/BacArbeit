import pandas as pd 
import sys

def main():
    df = pd.read_csv("load_database.csv")

    df = df.drop('top1', axis=1)
    df = df.drop('top5', axis=1)
    df = df.drop('map', axis=1)
    df = df.drop('inference type', axis=1)
    df = df.drop('os', axis=1)
    df = df.drop('miou', axis=1)
    df = df.drop("model_name", axis=1)

    df['api'] = df['api'].replace('onnx', 'ONNX')
    df['api'] = df['api'].replace('ov', 'OpenVINO')
    df['api'] = df['api'].replace('tf', 'TFLite')
    df['api'] = df['api'].replace('delegate', 'Arm NN Delegate')
    df['api'] = df['api'].replace('pytorch', 'PyTorch')

    df = df.rename(columns={
        "api": "Frameworks",
        "latency avg": "Mean Latency [s]",
        "load": "Load",
        "thread count": "Core Count"
    })

    load_mapping = {
    'default': 'No Load',
    'one': 'One Core Load',
    'two': 'Two Core Load',
    'three': 'Three Core Load'
}
    
    thread_mapping = {
        "False": "Default",
        "1": "One Core",
        "2": "Two Cores",
        "3": "Three Cores",
        "4": "Four Cores"
    }

    df['Load'] = df['Load'].replace(load_mapping)
    df["Core Count"] = df["Core Count"].replace(thread_mapping)

    df["Median Latency [s]"] = 0
    df["Standard Deviation [s]"] = 0 
    df["Variance [s²]"] = 0

    for i,r in df.iterrows():
        latency_link = r["latency"]

        #print(model_name, threads, api, latency_link)

        lat = pd.read_csv(latency_link)

        lat.to_csv("test.csv")

        median = lat["inference time"].median()
        var = lat["inference time"].var()
        std = lat["inference time"].std()

        df.at[i, "Median Latency [s]"] = median
        df.at[i, "Variance [s²]"] = var
        df.at[i, "Standard Deviation [s]"] = std



    df.to_csv("load_database_updated.csv")

if __name__ == "__main__":
    main()