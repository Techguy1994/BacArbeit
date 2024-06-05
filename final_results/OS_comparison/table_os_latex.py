import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import sys

def main():
    database = pd.read_csv("database_os_comparison.csv")
    csv = "temp_table_os.csv"

    df = create_empty_dataframe()

    df = interate_through_database(database, df)

    df.to_csv(csv)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    align='center'),
        cells=dict(values=[df.model_name, df.api, df.loc[:,"mean latency ubuntu"], df.loc[:,"mean latency raspberryos"]],
                align='center'), 
        )
    ])

    fig.show()



def create_empty_dataframe():
    dict = {
    "model_name": [],
    "api": [],
    "mean latency ubuntu": [],
    "mean latency raspberryos": []
}

    df = pd.DataFrame(dict)

    return df

def interate_through_database(database, df):
    for i,r in database.iterrows():
        model_name = r["model_name"]
        os = r["os"]
        api = r["api"]
        latency_link = r["latency"]

        index = df[(df['model_name'] == model_name)].index
        print(index.values)


        lat = pd.read_csv(latency_link)

        #print(lat.columns)

        #mean = lat.loc[:, "inference time"].mean()
        #var = lat.loc[:, "inference time"].var()
        mean = lat["inference time"].mean().round(3)

        if os == "ubuntus":
            mean_latency_str = "mean latency ubuntu"
        elif os == "raspberryos":
            mean_latency_str = "mean latency raspberryos"
        
        print(index.values)

        if index.values.size > 0:
            print(index.values)
            df.loc[index,[mean_latency_str]] = mean
        else:
            entry = {
                    "model_name": [model_name],
                    "api": [api],
                    mean_latency_str: [mean]
            }

            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True)

    return df 

if __name__ == "__main__":
    main()