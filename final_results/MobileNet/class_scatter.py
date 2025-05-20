import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns


def main():
    df = pd.read_csv("database_updated.csv")

    df = df[df['inference type'] != "seg"]
    df = df[df['inference type'] != "det"]

    

    db = db.drop('mIoU', axis=1)
    db = db.drop('mAP', axis=1)
    db = db.drop('inference type', axis=1)

    db.to_csv("db.csv")

    df = db

    df.to_csv("df.csv")

    mobileNetv2 = "MobileNetV2"
    mobileNetv3l = "MobileNetV3 Large"
    mobileNetv3s = "MobileNetV3 Small"

    top1 = "Top1"
    top5 = "Top5"

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
        y="Median Latency [s]",
        hue="Frameworks",
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