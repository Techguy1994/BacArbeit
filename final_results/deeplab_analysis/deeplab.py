import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

def main():
    database = pd.read_csv("database.csv")
    csv = "deeplabv3_.csv"

    database = database[database['inference type'] != "class"]
    db = database[database['inference type'] != "det"]

    db = db.drop('top1', axis=1)
    db = db.drop('top5', axis=1)
    db = db.drop('map', axis=1)
    db = db.drop('inference type', axis=1)
    db = db.drop('os', axis=1)

    db.to_csv("test.csv")

    df = db

    df['api'] = df['api'].replace('onnx', 'Onnx')
    df['api'] = df['api'].replace('ov', 'Openvino')
    df['api'] = df['api'].replace('tf', 'Tensorflow')
    df['api'] = df['api'].replace('delegate', 'Armnn Delegate')
    df['api'] = df['api'].replace('pytorch', 'PyTorch')

    df['model_name'] = df['model_name'].replace('deeplabv3_FP32', 'DeeplabV3 MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('deeplabv3_mobilenet_v3_large', 'DeeplabV3 MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('lite-model_deeplabv3-mobilenetv2_1_default_1', 'DeeplabV3 MobileNetV2 FP32')
    df['model_name'] = df['model_name'].replace('lite-model_deeplabv3-mobilenetv2-int8_1_default_1', 'DeeplabV3 MobileNetV2 INT8')

    

    df = df.rename(columns={
        "api": "API",
        "model_name": "Model"
    })

    df["miou"] = df["miou"] / 100

    df = df.sort_values(by='latency avg')

    # Extract Pareto front: increasing mIoU with increasing latency
    pareto_front = []
    max_miou = -float("inf")

    for _, row in df.iterrows():
        if row["miou"] > max_miou:
            pareto_front.append(row)
            max_miou = row["miou"]

    pareto_df = pd.DataFrame(pareto_front)

    

    # Plot with Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="miou",
        y="latency avg",
        hue="API",
        style="Model",
        s=200  # Marker size, adjust as needed
    )

    sns.lineplot(
    data=pareto_df,
    x="miou",
    y="latency avg",
    color="black",
    linewidth=2,
    marker="o",
    label="Pareto Front",
    alpha = 0.5,
    linestyle="--"
)

    plt.title("DeeplabV3")
    plt.xlabel("mIoU")
    plt.ylabel("Latency Avg")
    plt.legend(loc='upper left')  # Move legend out of the plot
    plt.tight_layout()
    plt.show()








if __name__ == "__main__":
    main()