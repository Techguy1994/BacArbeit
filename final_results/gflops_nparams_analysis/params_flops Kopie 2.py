import pandas as pd
import plotly.express as px
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    database = pd.read_csv("database_updated.csv")
    csv = "test.csv"

    df = create_empty_dataframe()

    df = interate_through_database(database, df)

    print(df)
    df.to_csv(csv)



# Set style for whitegrid
    sns.set(style="whitegrid")

    min_bubble_size = 50
    max_bubble_size = 2000

    plt.figure(figsize=(10, 6))

    # Plot
    ax = sns.scatterplot(
        data=df,
        x="GFLOPS",
        y="NParams",
        size="median_lat",
        sizes=(min_bubble_size, max_bubble_size),
        hue="Model",
        alpha=0.7,
        legend=False  # disable automatic legend
    )

    # Axis labels
    plt.grid(True)
    plt.xlabel("GFLOPS", fontsize=20)
    plt.ylabel("Number of Parameters [10⁶]", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # === Color legend (Model) ===
    from matplotlib.patches import Patch
    palette = sns.color_palette()  # or use ax._legend_data for more precision

    model_labels = df["Model"].unique()
    color_handles = [Patch(color=palette[i], label=model_labels[i]) for i in range(len(model_labels))]

    legend1 = plt.legend(
        handles=color_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=12,
        title_fontsize=14
    )
    plt.gca().add_artist(legend1)

    # === Size legend (Latency) ===
    import numpy as np
    from matplotlib.lines import Line2D

    # Pick a few representative latency values to display
    latency_values = [25, 50, 75, 100]
    size_handles = [
        Line2D([], [], linestyle="none", marker="o", alpha=0.5,
            markersize=np.sqrt(min_bubble_size + (max_bubble_size - min_bubble_size) * ((val - min(latency_values)) / (max(latency_values) - min(latency_values)))) / 2,
            color="gray", label=f"{val} ms")
        for val in latency_values
    ]

    legend2 = plt.legend(
        handles=size_handles,
        title="Median Latency",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=12,
        title_fontsize=14
    )

    plt.tight_layout()
    plt.savefig("flops.pdf", bbox_inches='tight')
    plt.show()


def create_empty_dataframe():
    dict = {
    "Model": [],
    "median_lat": [],
    "mAP": [],
    "GFLOPS": [],
    "NParams": []
    }

    df = pd.DataFrame(dict)

    return df



def interate_through_database(database, df):

    database = database[(database["Frameworks"] == "OpenVINO")]

    for i,r in database.iterrows():

        gflops_mparams_csv = "GFLOPS_MPARAMS.csv"
        gflops_mparams_df = pd.read_csv(gflops_mparams_csv, delimiter=";")

        print(gflops_mparams_df["model"])
        print(r["Model"])

        print(gflops_mparams_df[gflops_mparams_df["model"] == r["Model"]]["GFLOPS"])

        gflops = gflops_mparams_df[gflops_mparams_df["model"] == r["Model"]]["GFLOPS"].item()
        mparams = gflops_mparams_df[gflops_mparams_df["model"] == r["Model"]]["MParams"].item()

        entry = {
                "Model": r["Model"],
                "median_lat": [r["Median Latency [s]"]],
                "mAP": [r["mAP"]],
                "GFLOPS": [gflops],
                "NParams": [mparams]
            }
        
        df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 
    
    return df


if __name__ == "__main__":
    main()