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

    # Scale size values manually
    min_bubble_size = 50  # Minimum bubble size
    max_bubble_size = 2000  # Maximum bubble size

    print(df)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=df,
        x="GFlops",
        y="NParams in Million",
        size="mean_lat",  # Use scaled size
        sizes=(min_bubble_size, max_bubble_size),
        hue="model_name",
        alpha=0.7 # Adjust transparency
        #legend="full"  # Ensure full legend appears
    )

    # Customize the plot
    plt.grid(True)  # Enable grid
    plt.xlabel("GFlops", fontsize=20)
    plt.ylabel("NParams in Million", fontsize=20)
    #plt.title("", fontsize=24)

    plt.xticks(fontsize=12)  # Change x-axis tick size
    plt.yticks(fontsize=12)

    handles, labels = scatter.get_legend_handles_labels()
    num_models = len(df["model_name"].unique())  # Number of unique models

    # Keep the legend inside the graph on the **left**
    plt.legend(handles[1:num_models+1], labels[1:num_models+1], loc="upper left", frameon=True, title="Model", fontsize=12, title_fontsize=14) # Position legend inside on the 

    # Show the plot
    plt.show()

    #fig = px.scatter(df, x="GFlops", y="NParams in Million", size="map", color="model_name", hover_name="model_name", size_max=60)
    #fig.show()


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

        gflops = gflops_mparams_df[gflops_mparams_df["model"] == r["Model"]]["GFLOPS"].item()
        mparams = gflops_mparams_df[gflops_mparams_df["model"] == r["Model"]]["MParams"].item()

        entry = {
                "Model": r["Model"],
                "mean_lat": [r["Median Latency [s]"]],
                "mAP": [r["mAP"]],
                "GFlops": [gflops],
                "NParams": [mparams]
            }
        
        df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 
    
    return df


if __name__ == "__main__":
    main()