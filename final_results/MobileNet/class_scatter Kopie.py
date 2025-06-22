import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns


def main():
    df = pd.read_csv("database_updated.csv")

    df = df[df['inference type'] != "seg"]
    df = df[df['inference type'] != "det"]

    

    df = df.drop('mIoU', axis=1)
    df = df.drop('mAP', axis=1)
    df = df.drop('inference type', axis=1)

    

    mobileNetv2 = "MobileNetV2"
    mobileNetv3l = "MobileNetV3 Large"
    mobileNetv3s = "MobileNetV3 Small"

    top1 = "Top1"
    top5 = "Top5"

    model = mobileNetv3s
    top = top1

    if top == "Top1":
        df = df.drop("Top5", axis=1)
    elif top == "Top5":
        df = df.drop("Top1", axis=1)

    df = df[df["Model"].str.contains(model)]

    df = df.sort_values(by='Median Latency [s]')

    df.to_csv("df.csv")

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
        s=600  # Marker size, adjust as needed
    )

    sns.lineplot(
    data=pareto_df,
    x=top,
    y="Median Latency [s]",
    color="black",
    linewidth=2,
    marker="o",
    label="Pareto Front",
    alpha = 0.5,
    linestyle="--",
    markersize=10
)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.title(model, fontsize=24)
    plt.xlabel(top, fontsize=20)
    plt.ylabel("Median Latency [s]", fontsize=20)
    plt.legend(loc='upper left', fontsize=16)  # Move legend out of the plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(model + top + ".pdf", bbox_inches='tight')
    #plt.show()




if __name__ == "__main__":
    main()