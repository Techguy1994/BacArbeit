import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

def main():
    df = pd.read_csv("database_updated.csv")

    df = df[df['inference type'] != "class"]
    ddfb = df[df['inference type'] != "det"]

    df = df.drop('Top1', axis=1)
    df = df.drop('Top5', axis=1)
    df = df.drop('mAP', axis=1)
    df = df.drop('inference type', axis=1)


    df["mIoU"] = df["mIoU"] / 100

    df = df.sort_values(by='Median Latency [s]')

    # Extract Pareto front: increasing mIoU with increasing latency
    pareto_front = []
    max_miou = -float("inf")

    for _, row in df.iterrows():
        if row["mIoU"] > max_miou:
            pareto_front.append(row)
            max_miou = row["mIoU"]

    pareto_df = pd.DataFrame(pareto_front)

    

    # Plot with Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="mIoU",
        y="Median Latency [s]",
        hue="Frameworks",
        style="Model",
        s=200  # Marker size, adjust as needed
    )

    sns.lineplot(
    data=pareto_df,
    x="mIoU",
    y="Median Latency [s]",
    color="black",
    linewidth=2,
    marker="o",
    label="Pareto Front",
    alpha = 0.5,
    linestyle="--"
)

    plt.title("DeeplabV3")
    plt.xlabel("mIoU")
    plt.ylabel("Median Latency [s]")
    plt.legend(loc='upper left')  # Move legend out of the plot
    plt.tight_layout()
    plt.show()








if __name__ == "__main__":
    main()