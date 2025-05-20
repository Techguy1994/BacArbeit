import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd 
import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches




def main():
    database = pd.read_csv("load_database_updated.csv")
    df = create_empty_dataframe()

    db = database[database['Frameworks'] != "PyTorch"]

    unique_apis = db['Frameworks'].unique()
    unique_loads = db["Load"].unique()

    print(unique_apis, unique_loads)
    

    db = db.sort_values('Load')

    load_order = ['No Load', 'One Core Load', 'Two Core Load', 'Three Core Load']
    db['Load'] = pd.Categorical(db['Load'], categories=load_order, ordered=True)

    
    # 2. Fix the 'thread count' order
    thread_order = ['Default', 'One Core', 'Two Cores', 'Three Cores', 'Four Cores']
    db['Core Count'] = pd.Categorical(db['Core Count'], categories=thread_order, ordered=True)

    file_name = "bar_error_median.csv"
    
    """"
    if os.path.isfile(file_name):
        db = pd.read_csv(file_name)
    else:
        db = interate_through_database(db, df)
        db.to_csv(file_name)
    """

    db = db.sort_values('Load')

    fw_order = ["TFLite", "Arm NN Delegate", "ONNX", "OpenVINO"]
    db['Frameworks'] = pd.Categorical(db['Frameworks'], categories=fw_order, ordered=True)


    load_order = ['No Load', 'One Core Load', 'Two Core Load', 'Three Core Load']
    db['Load'] = pd.Categorical(db['Load'], categories=load_order, ordered=True)

    
    # 2. Fix the 'thread count' order
    thread_order = ['Default', 'One Core', 'Two Cores', 'Three Cores', 'Four Cores']
    db['Core Count'] = pd.Categorical(db['Core Count'], categories=thread_order, ordered=True)

    db.to_csv("t.csv")

    print(unique_apis)
    
    # Create the catplot with the correct y column
    g = sns.catplot(
        data=db,
        kind="bar",
        x="Load",
        y="Median Latency [s]",  # ← updated here
        hue="Core Count",
        col="Frameworks",
        col_wrap=2,
        palette="viridis",
        estimator=np.median,
        errorbar=None  # We'll add error bars manually
    )

    # Set y-axis limits, grid, and tick sizes
    for ax in g.axes.flat:
        ax.set_ylim(0, 0.3)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(False)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    # Axis labels and titles
    g.set_axis_labels("Artificial Load", "Median Latency (s)", fontsize=16)
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_title(ax.get_title(), fontsize=20)

    # Legend formatting
    g._legend.set_title("Core counts used for inference")
    g._legend.set_bbox_to_anchor((0.01, 1.1))
    g._legend.set_loc("upper left")
    for text in g._legend.get_texts():
        text.set_fontsize(13)
    g._legend.get_title().set_fontsize(15)

    # Global title
    plt.suptitle("MobileNetV3 Large", fontsize=24, y=1.1)

    # --- Add manual error bars using "Standard Deviation [s]" ---
    # Auto-detect the correct standard deviation column
    std_col = [col for col in db.columns if "Standard" in col and "Deviation" in col][0]

    for ax in g.axes.flat:
        framework = ax.get_title()  # Use facet title to get framework
        sub_db = db[db["Frameworks"] == framework]
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        
        for container in ax.containers:
            core_count = container.get_label()
            
            for bar_index, bar in enumerate(container):
                if not isinstance(bar, mpatches.Rectangle):
                    continue  # Skip non-bar elements

                load_index = bar_index // len(db["Core Count"].unique())
                if load_index >= len(xtick_labels):
                    continue
                load_label = xtick_labels[load_index]

                height = bar.get_height()
                x = bar.get_x() + bar.get_width() / 2

                match = sub_db[
                    (sub_db["Core Count"] == core_count) &
                    (sub_db["Load"] == load_label)
                ]

                if not match.empty:
                    err = match[std_col].values[0]
                    ax.errorbar(
                        x, height, yerr=err,
                        fmt='none', color='black', capsize=4, linewidth=2
                    )

    # Final layout and export
    plt.tight_layout()
    plt.savefig("bar_plot_4_subplot_error.pdf", bbox_inches='tight')



        


def create_empty_dataframe():
    dict = {
    "Latency [s]": [],
    "Core Count": [],
    "Frameworks": [],
    "Load": []
}

    df = pd.DataFrame(dict)

    return df
            

def interate_through_database(database, df):
    for i,r in database.iterrows():
        fw = r["Frameworks"]
        latency_link = r["latency"]
        threads = r["Core Count"]
        l = r["Load"]

        print(fw, threads, latency_link)

        lat = pd.read_csv(latency_link)

        for ii, rr in lat.iterrows():
            inference_time = rr["inference time"]

            entry = {"Frameworks": [fw],"Latency [s]": [inference_time], "Core Count": [threads], "Load": [l]}
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df







if __name__ == "__main__":
    main()
