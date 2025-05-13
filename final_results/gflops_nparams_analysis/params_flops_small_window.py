import pandas as pd
import plotly.express as px
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    database = pd.read_csv("database.csv")
    csv = "test.csv"

    df = create_empty_dataframe()

    df = interate_through_database(database, df)

    print(df)
    df.to_csv(csv)


    df['model_name'] = df['model_name'].replace('deeplabv3_FP32', 'DeeplabV3')
    df['model_name'] = df['model_name'].replace('mobilenet-v2-1.4-224_FP32', 'MobileNetV2')
    df['model_name'] = df['model_name'].replace('mobilenet-v3-large-1.0-224-tf_FP32', 'MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('mobilenet-v3-small-1.0-224-tf_FP32', 'MobileNetV3 Small')
    df['model_name'] = df['model_name'].replace('yolov5l', 'Yolov5l')
    df['model_name'] = df['model_name'].replace('yolov5m', 'Yolov5m')
    df['model_name'] = df['model_name'].replace('yolov5n', 'Yolov5n')
    df['model_name'] = df['model_name'].replace('yolov5s', 'Yolov5s')

    df = df[df['model_name'] != "Yolov5l"]
    df = df[df['model_name'] != "Yolov5m"]
    


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

    plt.xlim(0, 20)  # Set x-axis range (adjust as needed)
    plt.ylim(0, 8)

    handles, labels = scatter.get_legend_handles_labels()
    num_models = len(df["model_name"].unique())  # Number of unique models

    # Keep the legend inside the graph on the **left**
    plt.legend(handles[1:num_models+1], labels[1:num_models+1], loc="upper center", frameon=True, title="Model", fontsize=12, title_fontsize=14) # Position legend inside on the 

    # Show the plot
    plt.show()

    #fig = px.scatter(df, x="GFlops", y="NParams in Million", size="map", color="model_name", hover_name="model_name", size_max=60)
    #fig.show()


def create_empty_dataframe():
    dict = {
    "model_name": [],
    "mean_lat": [],
    "map": [],
    "GFlops": [],
    "NParams in Million": []
    }

    df = pd.DataFrame(dict)

    return df



def interate_through_database(database, df):

    database = database[(database["api"] == "ov")]

    for i,r in database.iterrows():
        #print(r["model_name"])

        gflops_mparams_csv = "GFLOPS_MPARAMS.csv"
        gflops_mparams_df = pd.read_csv(gflops_mparams_csv, delimiter=";")

        print(r["model_name"])

        print(gflops_mparams_df[gflops_mparams_df["model"] == r["model_name"]]["GFLOPS"])

        gflops = gflops_mparams_df[gflops_mparams_df["model"] == r["model_name"]]["GFLOPS"].item()
        mparams = gflops_mparams_df[gflops_mparams_df["model"] == r["model_name"]]["MParams"].item()

        entry = {
                "model_name": r["model_name"],
                "mean_lat": [r["latency avg"]],
                "map": [r["map"]],
                "GFlops": [gflops],
                "NParams in Million": [mparams]
            }
        
        df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 
    
    return df


if __name__ == "__main__":
    main()