import pandas as pd 
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import os
import numpy as np

def main():
    database = pd.read_csv("database.csv")

    database = database[database['inference type'] != "class"]
    db = database[database['inference type'] != "seg"]

    db = db.drop('top1', axis=1)
    db = db.drop('top5', axis=1)
    db = db.drop('miou', axis=1)
    db = db.drop('inference type', axis=1)
    db = db.drop('os', axis=1)

    db.to_csv("db.csv")



    #database = database.sort_values("latency avg")

    df = db

    df = df.rename(columns={
        "api": "API",
        "model_name": "Model",
        "latency avg": "average latency"
    })

    
    df['API'] = df['API'].replace('onnx', 'Onnx')
    df['API'] = df['API'].replace('ov', 'Openvino')
    df['API'] = df['API'].replace('tf', 'Tensorflow')
    df['API'] = df['API'].replace('delegate', 'Armnn Delegate')
    df['API'] = df['API'].replace('pytorch', 'PyTorch')

    df.to_csv("df.csv")

    df = df.sort_values(by='average latency')

    pareto_front = []
    max_top = -float("inf")

    for _, row in df.iterrows():
        if row["map"] > max_top:
            pareto_front.append(row)
            max_top = row["map"]

    pareto_df = pd.DataFrame(pareto_front)

    

    # Plot with Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="map",
        y="average latency",
        hue="API",
        style="Model",
        s=200  # Marker size, adjust as needed
    )

    sns.lineplot(
    data=pareto_df,
    x="map",
    y="average latency",
    color="black",
    linewidth=2,
    marker="o",
    label="Pareto Front",
    alpha = 0.5,
    linestyle="--"
)

    plt.title("Yolo")
    plt.xlabel("map")
    plt.ylabel("Latency Avg")
    plt.legend(loc='upper left')  # Move legend out of the plot
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()