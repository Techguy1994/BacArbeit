import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import sys
import os


def main():
    database = pd.read_csv("database.csv")
    print(database)

    

    df = create_empty_dataframe()
    print(df.head)


   

    df = interate_through_database(database, df)

    print(df)
    df.to_csv("temp.csv")

    fu = ["fp32", "uint8"]

    mask = df['model_name'].str.contains(fu[1])
    df = df[mask]


    #fig = px.box(df, x="model_name", y="latency", color="api", range_y=[0.005,0.1])


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
    fig.update_layout(violingap=0, violinmode='overlay')
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
