import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    database = pd.read_csv("database_updated.csv")
    print(database)

    database = database[database['inference type'] != "seg"]
    database = database[database['inference type'] != "det"]

    

    df = create_empty_dataframe()
    print(df.head)

    df = interate_through_database(database, df)

    print(df)
    df.to_csv("temp.csv")


    #fu = ["FP32", "INT8"]

    #print(df["Model"].unique())

    #choice = 0

    #mask = df['Model'].str.contains(fu[choice])
    #df = df[mask]


    #fig = px.box(df, x="model_name", y="latency", color="api", range_y=[0.005,0.1])

    #if choice == 0:
    #    y_range = [0.01,0.06]
    #elif choice == 1:
    #    y_range = [0, 0.035]




    fig = go.Figure()

    fig.add_trace(go.Violin(x=df['Model'][ df['Framework'] == 'TFLite' ],
                            y=df['latency'][ df['Framework'] == 'TFLite' ],
                            legendgroup='Yes', scalegroup='Yes', name='TFLite',
                            side='negative',
                            line_color='blue')
                )
    fig.add_trace(go.Violin(x=df['Model'][ df['Framework'] == 'Arm NN Delegate' ],
                            y=df['latency'][ df['Framework'] == 'Arm NN Delegate' ],
                            legendgroup='No', scalegroup='No', name='Armnn NN Delegate',
                            side='positive',
                            line_color='orange')
                )
    fig.update_traces(meanline_visible=True)


    fig.update_layout(
    violingap=0, violinmode='overlay',
    plot_bgcolor='white',
    autosize = True,
    margin=dict(l=50, r=300, t=50, b=50),
    font=dict(family="Arial", size=14), # White background
    xaxis=dict(
        showgrid=True, gridcolor='lightgray',  # Show gridlines on x-axis
        zeroline=True, zerolinecolor='gray', title =dict(text = "Model ", font=dict(size=24)),
        tickfont = dict(size=16)  # Add zero line
    ),
    yaxis=dict(
        showgrid=True, gridcolor='lightgray',  # Show gridlines on y-axis
        zeroline=True, zerolinecolor='gray', title = dict(text = "Latency (s)", font=dict(size=24)), tickfont=dict(size=16)
    ),
    legend = dict(font=dict(size=20))
)
    fig.show()




def create_empty_dataframe():
    dict = {
    "Model": [],
    "Framework": [],
    "latency": []
}

    df = pd.DataFrame(dict)

    return df


def interate_through_database(database, df):
    for i,r in database.iterrows():

        if r["Frameworks"] in ["Arm NN Delegate","TFLite"]:
            model_name = r["Model"]
            api = r["Frameworks"]
            latency_link = r["latency"]

            lat = pd.read_csv(latency_link)

            for ii, rr in lat.iterrows():
                inference_time = rr["inference time"]

                entry = {"Model": [model_name], "Framework": [api], "latency": [inference_time]}
                df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df




if __name__ == "__main__":
    main()
