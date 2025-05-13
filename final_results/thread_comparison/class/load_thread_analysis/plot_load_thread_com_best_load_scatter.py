import pandas as pd 
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines 

def main():
    database = pd.read_csv("load_database.csv")
    df = create_empty_dataframe()

    db = database[database['api'] != "pytorch"]

    unique_apis = db['api'].unique()
    unique_loads = db["load"].unique()

    sort_loads = ["No load", "1 thread load", "2 thread load", "3 thread load"]



    for api in unique_apis:
        for load in unique_loads:
            filtered_db = db[(db["api"] == api) & (db["load"] == load)]
            max_value = filtered_db["latency avg"].min()
            #print(max_value, api, load)

            filtered_max_db = filtered_db[filtered_db["latency avg"] == max_value]

            for i,r in filtered_max_db.iterrows():
                threads = r["thread count"]

            if load == "default":
                load = "No load"
            elif load == "one":
                load = "1 thread load"
            elif load == "two":
                load = "2 thread load"
            elif load == "three":
                load = "3 thread load"

            if threads == "False":
                threads = "Default"
           
            entry = {"mean latency": [max_value], "thread": [threads], "api": [api], "load": [load]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True)
    
    df['load'] = pd.Categorical(df['load'], categories=sort_loads, ordered=True)
    df = df.sort_values('load')

    df['api'] = df['api'].replace('onnx', 'ONNX')
    df['api'] = df['api'].replace('ov', 'OpenVINO')
    df['api'] = df['api'].replace('tf', 'Tensorflow')
    df['api'] = df['api'].replace('delegate', 'ArmNN Delegate')

    

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
        data=df, x="load", y="mean latency", hue="api", style="thread", 
        s=250, ax=ax  # Adjust marker size
    )

    # Add line plots for each 'api'
    for api in df['api'].unique():
        color_df = df[df['api'] == api]
        ax.plot(
            color_df['load'], color_df['mean latency'], 
            linestyle='-', linewidth=4, alpha=0.25  # Adjust line width
        )

    # Labels and title
    #ax.set_title('Comparison for different CPU loads for MobileNet V3 Large', fontsize=16)  # Title size
    ax.set_xlabel('CPU load on different amount of cores', fontsize=20)  # X-axis label size
    ax.set_ylabel('Mean latency (s)', fontsize=20)  # Y-axis label size
    ax.set_ylim(0, 0.2) 

    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Legend adjustments
    handles, labels = ax.get_legend_handles_labels()

    manual_order = [
    "Frameworks",  # Placeholder for API section
    "ONNX", "Tensorflow", "OpenVINO", "Arm Delegate", # Example API names (replace with actual ones from df)
    "Thread count",  # Placeholder for Thread count section
    "Default", "1", "2", "3", "4"
    ]

    # Sort handles according to manual_order

    section_handle = mlines.Line2D([], [], color="white", label=" ")  # Invisible legend item

    new_handles = []
    new_labels = []

    for label in manual_order:
        if label in labels:
            new_handles.append(handles[labels.index(label)])
            new_labels.append(label)
        elif label in ["Frameworks", "Thread count"]:  # Add section headers
            new_handles.append(section_handle)
            new_labels.append(label)


    #new_labels = ["API" if label == "api" else "Thread count" if label == "thread" else label for label in labels]
    ax.legend(new_handles, new_labels, title="Frameworks and the amount of threads set", fontsize=13, title_fontsize=15,loc='upper left')

    plt.tight_layout()
    plt.savefig("scatter_comp_updated.pdf", bbox_inches='tight')
    

    # Show the plot
    plt.savefig("scatter_comp_updated.png", dpi=300, bbox_inches='tight')


        


def create_empty_dataframe():
    dict = {
    "mean latency": [],
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
