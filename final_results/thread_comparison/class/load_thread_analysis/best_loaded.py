import pandas as pd 
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines 
from matplotlib.lines import Line2D

def main():
    database = pd.read_csv("load_database_updated.csv")
    df = create_empty_dataframe()

    db = database[database['Frameworks'] != "PyTorch"]

    unique_apis = db['Frameworks'].unique()
    unique_loads = db["Load"].unique()



    sort_loads = ["No Load", "One Core Load", "Two Core Load", "Three Core Load"]



    for api in unique_apis:
        for load in unique_loads:

            filtered_db = db[(db["Frameworks"] == api) & (db["Load"] == load)]
            max_value = filtered_db["Median Latency [s]"].min()
            #print(max_value, api, load)

            filtered_max_db = filtered_db[filtered_db["Median Latency [s]"] == max_value]

            for i,r in filtered_max_db.iterrows():
                threads = r["Core Count"]
           
            entry = {"median latency": [max_value], "thread": [threads], "api": [api], "load": [load]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True)
    
    df['load'] = pd.Categorical(df['load'], categories=sort_loads, ordered=True)
    df = df.sort_values('load')


    

    df.to_csv("scatter.csv")

    # start of the scatter plot



    # Sample DataFrame structure (replace this with your actual df)
    # df = pd.read_csv("your_data.csv")

    # Set seaborn style


    sns.set(style="whitegrid")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))  # Adjust figure size

    # Scatter plot with different colors for each 'api'
    sns.scatterplot(
        data=df, x="load", y="median latency", hue="api", style="thread", 
        s=250, ax=ax  # Adjust marker size
    )

    # Add line plots for each 'api'
    for api in df['api'].unique():
        color_df = df[df['api'] == api]
        ax.plot(
            color_df['load'], color_df['median latency'], 
            linestyle='-', linewidth=4, alpha=0.25  # Adjust line width
        )

    # Labels and title
    ax.set_xlabel('CPU load on different amount of cores', fontsize=20)  # X-axis label size
    ax.set_ylabel('Median latency [s]', fontsize=20)  # Y-axis label size
    ax.set_ylim(0, 0.2)

    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Legend adjustments
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    # Manual groupings
    framework_labels = ["TFLite", "Arm NN Delegate", "OpenVINO", "ONNX"]
    corecount_labels = ["Default", "One Core", "Two Cores", "Three Cores", "Four Cores"]

    # Framework legend
    framework_handles =  [label_to_handle[l] for l in framework_labels if l in label_to_handle]
    framework_labels_display = [l for l in framework_labels if l in label_to_handle]

    # Core count legend
    corecount_handles = [label_to_handle[l] for l in corecount_labels if l in label_to_handle]
    corecount_labels_display = [l for l in corecount_labels if l in label_to_handle]

    # Remove default legend
    ax.legend_.remove()
    ax.set_title("MobileNetV3 Large", fontsize=24)

    # Add the legends side by side
    legend1 = ax.legend(framework_handles, framework_labels_display, loc='upper left',
                        bbox_to_anchor=(0.01, 0.99), fontsize=15, title_fontsize=20, title = "Frameworks", frameon=True)

    legend2 = ax.legend(corecount_handles, corecount_labels_display, loc='upper left',
                        bbox_to_anchor=(0.25, 0.99), fontsize=15, title_fontsize=20, title="Core counts used for inference", frameon=True)

    ax.add_artist(legend1)
    ax.add_artist(legend2)

    # Save
    plt.tight_layout()
    plt.savefig("scatter_comp_loaded.pdf", bbox_inches='tight')

    

    # Show the plot
    #plt.savefig("scatter_comp_updated.png", dpi=300, bbox_inches='tight')


        


def create_empty_dataframe():
    dict = {
    "median latency": [],
    "thread": [],
    "api": [],
    "load": []
}

    df = pd.DataFrame(dict)

    return df
            



def interate_through_database(database, df):
    for i,r in database.iterrows():
        avg = r["latency avg"]
        latency_link = r["latency"]
        threads = r["thread count"]
        api = r["api"]
        l = r["load"]
        print(l)
        if l == "default":
            print(l)
            l = "noload"

        #print(avg, threads, api)

        lat = pd.read_csv(latency_link)

        for ii, rr in lat.iterrows():
            inference_time = rr["inference time"]

            entry = {"mean latency": [avg],"latency": [inference_time], "thread": [threads], "api": [api], "load": [l]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df



if __name__ == "__main__":
    main()
