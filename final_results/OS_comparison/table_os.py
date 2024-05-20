import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 

def main():
    database = pd.read_csv("database_os_comparison.csv")
    csv = "temp_table_os.csv"

    df = create_empty_dataframe()

    df = interate_through_database(database, df)

    df.to_csv(csv)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df.model_name, df.os, df.api, df.loc[:,"mean latency"], df.loc[:,"var latency"], df.loc[:,"90% percent"], df.loc[:,"99% percent"]],
                fill_color='lavender',
                align='left'))
    ])

    fig.show()



def create_empty_dataframe():
    dict = {
    "model_name": [],
    "os": [],
    "api": [],
    "mean latency": [],
    "var latency": [],
    "90% percent": [],
    "99% percent": []
}

    df = pd.DataFrame(dict)

    return df

def interate_through_database(database, df):
    for i,r in database.iterrows():
        model_name = r["model_name"]
        os = r["os"]
        api = r["api"]
        latency_link = r["latency"]

        lat = pd.read_csv(latency_link)

        #print(lat.columns)

        #mean = lat.loc[:, "inference time"].mean()
        #var = lat.loc[:, "inference time"].var()
        mean = lat["inference time"].mean()
        var = lat["inference time"].var()
        ninety_percentile = lat["inference time"].quantile(q=0.90)
        ninetynine_percenitle = lat["inference time"].quantile(q=0.99)

        entry = {
                "model_name": [model_name],
                "os": [os],
                "api": [api],
                "mean latency": [mean],
                "var latency": [var],
                "90% percent": [ninety_percentile],
                "99% percent": [ninetynine_percenitle]
        }

        df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True)

    return df 

if __name__ == "__main__":
    main()