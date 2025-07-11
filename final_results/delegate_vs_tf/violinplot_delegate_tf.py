import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.io as pio   
pio.kaleido.scope.mathjax = None


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

    """
    df['Model'] = df['Model'].replace('MobileNetV2', 'V2')
    df['Model'] = df['Model'].replace('MobileNetV2 INT8', "V2 INT8")
    df['Model'] = df['Model'].replace('MobileNetV3 Large', "V3 Large")
    df['Model'] = df['Model'].replace('MobileNetV3 Large INT8', "V3 Large INT8")
    df['Model'] = df['Model'].replace('MobileNetV3 Small', "V3 Small")
    df['Model'] = df['Model'].replace('MobileNetV3 Small INT8', "V3 Small INT8")
    """


    df_clean = remove_outliers_percentile(df, group_cols=["Model", "Framework"], value_col="latency")

    df = df_clean

    df_int8 = df[df['Model'].str.contains("INT8")]

# Non-INT8 models only
    df_non_int8 = df[~df['Model'].str.contains("INT8")]
    
    range_int8 = []
    rannge_non_int8 = []

    text_non_int8 = "MobileNet Model FP32" 
    text_int8 = "MobileNet Model INT8" 

    df = df_int8


    fig = go.Figure()

    fig.add_trace(go.Violin(x=df['Model'][ df['Framework'] == 'TFLite' ],
                            y=df['latency'][ df['Framework'] == 'TFLite' ],
                            legendgroup='Yes', scalegroup='Yes', name='TFLite',
                            side='negative',
                            line_color='blue',
                            width = 1)
                )
    fig.add_trace(go.Violin(x=df['Model'][ df['Framework'] == 'Arm NN Delegate' ],
                            y=df['latency'][ df['Framework'] == 'Arm NN Delegate' ],
                            legendgroup='No', scalegroup='No', name='Armnn NN Delegate',
                            side='positive',
                            line_color='orange',
                            width = 1)
                )
    fig.update_traces(meanline_visible=True)


    fig.update_layout(
    violingap=0, violinmode='group',
    plot_bgcolor='white',
    autosize = True,
    margin=dict(l=0, r=50, t=0, b=0),
    font=dict(family="Arial", size=14), # White background
    xaxis=dict(
        showgrid=True, gridcolor='lightgray',  # Show gridlines on x-axis
        zeroline=True, zerolinecolor='gray', title =dict(text = text_int8, font=dict(size=20)),
        tickfont = dict(size=16)  # Add zero line
    ),
    yaxis=dict(
        showgrid=True, gridcolor='lightgray',  # Show gridlines on y-axis
        zeroline=True, zerolinecolor='gray', title = dict(text = "Latency [s]", font=dict(size=20)), tickfont=dict(size=16)
    ),


    legend=dict(
        x=0.98,  # Near right edge
        y=0.98,  # Near top
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.7)',
        bordercolor='gray',
        borderwidth=1,
        font=dict(size=16)
    )
    )

    fig.write_image("violin_latency.pdf", engine="kaleido")


    #fig.update_traces(width=1)  # narrower violins

    #fig.write_image("violin_latency.svg")
    #fig.write_image("violin_latency.pdf")

    fig.show()


def remove_outliers_percentile(df, group_cols, value_col, lower_pct=0.01, upper_pct=0.99):
    def filter_group(group):
        lower = group[value_col].quantile(lower_pct)
        upper = group[value_col].quantile(upper_pct)
        return group[(group[value_col] >= lower) & (group[value_col] <= upper)]
    
    return df.groupby(group_cols, group_keys=False).apply(filter_group)

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
