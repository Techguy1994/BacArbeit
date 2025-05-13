import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    database = pd.read_csv("database.csv")
    print(database)

    

    df = create_empty_dataframe()
    print(df.head)


   

    df = interate_through_database(database, df)

    print(df)
    df.to_csv("temp.csv")

    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v2_100_224_fp32_1', 'MobileNetV2 FP32')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v2_100_224_uint8_1', 'MobileNetV2 UINT8')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v3_large_100_224_fp32_1', 'MobileNetV3 Large FP32')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v3_large_100_224_uint8_1', 'MobileNetV3 Large UINT8')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v3_small_100_224_fp32_1', 'MobileNetV3 Small FP32')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v3_small_100_224_uint8_1', 'MobileNetV3 Small UINT8')



    fu = ["FP32", "UINT8"]

    print(df["model_name"].unique())

    choice = 1

    mask = df['model_name'].str.contains(fu[choice])
    df = df[mask]


    #fig = px.box(df, x="model_name", y="latency", color="api", range_y=[0.005,0.1])

    if choice == 0:
        y_range = [0.01,0.06]
    elif choice == 1:
        y_range = [0, 0.035]




    fig = go.Figure()

    fig.add_trace(go.Violin(x=df['model_name'][ df['api'] == 'tf' ],
                            y=df['latency'][ df['api'] == 'tf' ],
                            legendgroup='Yes', scalegroup='Yes', name='Tensorflite Runtime',
                            side='negative',
                            line_color='blue')
                )
    fig.add_trace(go.Violin(x=df['model_name'][ df['api'] == 'delegate' ],
                            y=df['latency'][ df['api'] == 'delegate' ],
                            legendgroup='No', scalegroup='No', name='Armnn Delegate',
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
        zeroline=True, zerolinecolor='gray', title =dict(text = "Model " + fu[choice], font=dict(size=24)),
        tickfont = dict(size=16)  # Add zero line
    ),
    yaxis=dict(
        showgrid=True, gridcolor='lightgray',  # Show gridlines on y-axis
        zeroline=True, zerolinecolor='gray', title = dict(text = "Latency (s)", font=dict(size=24)), range=y_range, tickfont=dict(size=16)
    ),
    legend = dict(font=dict(size=20))
)
    fig.show()




def create_empty_dataframe():
    dict = {
    "model_name": [],
    "api": [],
    "latency": []
}

    df = pd.DataFrame(dict)

    return df

def filter_through_database(df):
    pass
    #print(df.columns.values)
    #filt = df["api"] == "tf"
    #print(filt)
    #print(df.loc[filt])

def interate_through_database(database, df):
    for i,r in database.iterrows():

        if r["api"] in ["delegate","tf"]:
            if r["os"] == "ubuntus":
                model_name = r["model_name"]
                os = r["os"]
                api = r["api"]
                latency_link = r["latency"]

                lat = pd.read_csv(latency_link)

                for ii, rr in lat.iterrows():
                    inference_time = rr["inference time"]

                    entry = {"model_name": [model_name], "api": [api], "latency": [inference_time]}
                    df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 

    return df




if __name__ == "__main__":
    main()
