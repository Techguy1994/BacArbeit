import pandas as pd
import plotly.express as px
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def main():
    database = pd.read_csv("database_updated.csv")
    csv = "test.csv"

    df = create_empty_dataframe()

    df = interate_through_database(database, df)

    print(df)
    df.to_csv(csv)

    df['Model'] = df['Model'].replace('DeeplabV3 MobileNetV3 Large', 'DeeplabV3')



    df_clean = df.dropna(subset=["median_lat", "GFLOPS", "NParams"])

    # Bubble size scaling
    min_bubble_size = 50
    max_bubble_size = 2000
    lat_min = df_clean["median_lat"].min()
    lat_max = df_clean["median_lat"].max()
    df_clean["bubble_size"] = min_bubble_size + (df_clean["median_lat"] - lat_min) / (lat_max - lat_min) * (max_bubble_size - min_bubble_size)

    # Color palette mapping for models
    unique_models = df_clean["Model"].unique()
    palette = sns.color_palette(n_colors=len(unique_models))
    color_map = dict(zip(unique_models, palette))

    # Create the bubble plot
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=df_clean,
        x="GFLOPS",
        y="NParams",
        size="median_lat",
        sizes=(min_bubble_size, max_bubble_size),
        hue="Model",
        palette=color_map,
        alpha=0.7,
        legend=False  # We'll add a custom legend
    )

    # Axis labels
    plt.grid(True)
    plt.xlabel("GFLOPS", fontsize=18)
    plt.ylabel("Number of Parameters [10⁶]", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlim(0, 20)  # Set x-axis range (adjust as needed)
    plt.ylim(0, 8)

    # Custom legend: exact bubble size & latency for each model
    df_sorted = df_clean.sort_values(by="median_lat", ascending=False)

    # Create custom legend handles from sorted data
    custom_handles = [
        Line2D(
            [], [], linestyle="none", marker="o",
            markersize=np.sqrt(row["bubble_size"])/2,
            label=f"{row['Model']} ({(row['median_lat']):.2f} s)",
            color=color_map[row["Model"]]
        )
        for _, row in df_sorted.iterrows()
    ]

    print(custom_handles)

    custom_handles = custom_handles[2:]
    print(custom_handles)

    # Add the legend to the right of the plot
    plt.legend(
        handles=custom_handles,
        title="Model (Median Latency [s])",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.01),
        fontsize=12,
        title_fontsize=14,
        labelspacing=1
    )

    plt.tight_layout()
    plt.savefig("flops_custom_all_sizes_zoomed.pdf", bbox_inches='tight')
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