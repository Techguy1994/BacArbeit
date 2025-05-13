import pandas as pd 
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    database = pd.read_csv("database.csv")

    database = database[database['inference type'] != "class"]
    db = database[database['inference type'] != "seg"]

    db = db.drop('top1', axis=1)
    db = db.drop('top5', axis=1)
    db = db.drop('miou', axis=1)
    db = db.drop('inference type', axis=1)
    db = db.drop('os', axis=1)

    db.to_csv("test.csv")

    #database = database.sort_values("latency avg")
    df = db
    df = df.rename(columns={
        "api": "API",
        "model_name": "Model"
    })

    df['API'] = df['API'].replace('onnx', 'Onnx')
    df['API'] = df['API'].replace('ov', 'Openvino')
    df['API'] = df['API'].replace('tf', 'Tensorflow')
    df['API'] = df['API'].replace('delegate', 'Armnn Delegate')
    df['API'] = df['API'].replace('pytorch', 'PyTorch')

    df.to_csv("temp.csv")

    # Clean the DataFrame
    df_clean = df[["Model", "API", "latency avg", "map"]].copy()
    df_clean.rename(columns={"latency avg": "Latency", "map": "mAP", "API": "Framework"}, inplace=True)

    # Sort models to maintain consistent order
    model_order = ["yolov5n", "yolov5s", "yolov5m", "yolov5l"]
    df_clean["Model"] = pd.Categorical(df_clean["Model"], categories=model_order, ordered=True)
    df_clean.sort_values(["Model", "Framework"], inplace=True)

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Barplot for latency (left y-axis)
    barplot = sns.barplot(
        data=df_clean, x="Model", y="Latency", hue="Framework", ax=ax1,
        palette="pastel", dodge=True
    )
    ax1.set_ylabel("Latency (ms)", color="black")
    ax1.set_xlabel("YOLOv5 Model")
    ax1.set_title("Latency and mAP by YOLOv5 Model and Framework")

    # Create second axis for mAP
    ax2 = ax1.twinx()

    # Overlay mAP as points
    for i, (framework, group) in enumerate(df_clean.groupby("Framework")):
        offsets = {"tf": -0.25, "onnx": 0.0, "torchscript": 0.25}
        offset = offsets.get(framework, 0)
        x_vals = [model_order.index(m) + offset for m in group["Model"]]
        ax2.plot(x_vals, group["mAP"], 'o-', label=f"{framework} mAP", linewidth=2)

    ax2.set_ylabel("mAP")
    ax2.set_ylim(0, 0.5)

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()