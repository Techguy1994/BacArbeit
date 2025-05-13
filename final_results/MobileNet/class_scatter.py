import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns


def main():
    database = pd.read_csv("database.csv")
    csv = "deeplabv3_.csv"

    database = database[database['inference type'] != "seg"]
    db = database[database['inference type'] != "det"]

    db = db[db["os"].str.contains("raspberryos")==False]
    db = db[db["model_name"].str.contains("mobilenetv2-12-int8")==False]
    db = db[db["model_name"].str.contains("mobilenetv3_small_075_Opset17")==False]
    db = db[db["model_name"].str.contains("mobilenet_v3_large_q")==False]
    db = db[db["model_name"].str.contains("mobilenet_v2_q")==False]
    

    db = db.drop('miou', axis=1)
    db = db.drop('map', axis=1)
    db = db.drop('inference type', axis=1)
    db = db.drop('os', axis=1)

    db.to_csv("db.csv")

    

    df = db

    df['api'] = df['api'].replace('onnx', 'Onnx')
    df['api'] = df['api'].replace('ov', 'Openvino')
    df['api'] = df['api'].replace('tf', 'Tensorflow')
    df['api'] = df['api'].replace('delegate', 'Armnn Delegate')
    df['api'] = df['api'].replace('pytorch', 'PyTorch')


    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v2_100_224_fp32_1', 'MobileNetV2')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v2_100_224_uint8_1', 'MobileNetV2 INT8')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v3_large_100_224_fp32_1', 'MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v3_large_100_224_uint8_1', 'MobileNetV3 Large INT8')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v3_small_100_224_fp32_1', 'MobileNetV3 Small')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v3_small_100_224_uint8_1', 'MobileNetV3 Small INT8')

    df['model_name'] = df['model_name'].replace('mobilenetv2-12', 'MobileNetV2')
    df['model_name'] = df['model_name'].replace('mobilenetv3_large_100_Opset17', 'MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('mobilenetv3_small_050_Opset17', 'MobileNetV3 Small')

    df['model_name'] = df['model_name'].replace('mobilenet_v2', 'MobileNetV2')
    df['model_name'] = df['model_name'].replace('mobilenet_v3_large', 'MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('mobilenet_v3_small', 'MobileNetV3 Small')

    df['model_name'] = df['model_name'].replace('mobilenet-v2-1.4-224_FP32', 'MobileNetV2')
    df['model_name'] = df['model_name'].replace('mobilenet-v3-large-1.0-224-tf_FP32', 'MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('mobilenet-v3-small-1.0-224-tf_FP32', 'MobileNetV3 Small')

    df = df.rename(columns={
        "api": "API",
        "model_name": "Model",
        "top1": "Top 1",
        "top5": "Top 5",
        "latency avg": "average latency"
    })

    df.to_csv("df.csv")

    mobileNetv2 = "MobileNetV2"
    mobileNetv3l = "MobileNetV3 Large"
    mobileNetv3s = "MobileNetV3 Small"

    top1 = "Top 1"
    top5 = "Top 5"

    model = mobileNetv3s
    top = top1

    df = df[df["Model"].str.contains(mobileNetv2)==False]
    df = df[df["Model"].str.contains(mobileNetv3l)==False]

    df = df.sort_values(by='average latency')

    # Extract Pareto front: increasing mIoU with increasing latency
    pareto_front = []
    max_top = -float("inf")

    for _, row in df.iterrows():
        if row[top] > max_top:
            pareto_front.append(row)
            max_top = row[top]

    pareto_df = pd.DataFrame(pareto_front)

    

    # Plot with Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x=top,
        y="average latency",
        hue="API",
        style="Model",
        s=200  # Marker size, adjust as needed
    )

    sns.lineplot(
    data=pareto_df,
    x=top,
    y="average latency",
    color="black",
    linewidth=2,
    marker="o",
    label="Pareto Front",
    alpha = 0.5,
    linestyle="--"
)

    plt.title(model)
    plt.xlabel(top)
    plt.ylabel("Latency Avg")
    plt.legend(loc='upper left')  # Move legend out of the plot
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()