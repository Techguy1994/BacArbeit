import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import sys

def main():
    database = pd.read_csv("database.csv")
    csv = "deeplabv3.csv"

    df = create_empty_dataframe()

    bool = False

    df = interate_through_database(database, df)
    df.to_csv(csv)
    
    if bool:
        df = df.sort_values(by=['Average latency', 'MIoU'])
        print(df)
        df["MIoU"] = df["MIoU"] / 100

        #pareto_points_top1 = pareto_top1(df)
        print(df)
        
        #top1 plot
        fig = px.scatter(df, x="MIoU", y="Average latency", color="API", symbol="Model", title="DeeplabV3")
        fig.update_traces(marker=dict(size=20),
                    selector=dict(mode='markers'))
        #fig.add_trace(
        #    go.Scatter(
        #        x=pareto_points_top1['top1'],
        #        y=pareto_points_top1['mean_lat'],
        #        mode='lines+markers',
        #        name='Pareto Front',
        #        line=dict(color='red', width=2),
        #        marker=dict(size=8)
        #)
    #)
        fig.show()
    else:
        pass





def pareto_top(df):
    print("Df")
    print(df)
    
    pareto_df = df[['Average latency', 'top1']]

    print("just lat and top1")
    print(pareto_df)


    pareto_indices = is_pareto_efficient(pareto_df.to_numpy())
    pareto_points = pareto_df[pareto_indices]

    return pareto_points






def is_pareto_efficient(costs):

    print("pareto function df")
    print(costs)
    print(costs.shape)
    
    

    is_efficient_bool = np.ones(costs.shape[0], dtype=bool)
    is_efficient = np.ones(costs.shape[0])
    k = 3

    optimal = np.array([0,1]).reshape(1,2)
    print("------")
    print("optimal vector")
    print(optimal)
    print(optimal.shape)
    



    for i in range(costs.shape[0]):
        print("next pass")
        print(costs[i][0], costs[i][1])

        v_diff = optimal - costs[i,:] 

        print(v_diff)
        is_efficient[i] = np.sqrt((v_diff[0,0]*1).round(3)**2 + v_diff[0,1].round(3)**2)
        print(v_diff)
        print(is_efficient[i])
       

    
    
    print(is_efficient)
    
    idx = np.argpartition(is_efficient, k)
    print(idx)

    lowest_indices = np.argsort(is_efficient)[:3]
    print(lowest_indices)

    for i in range(idx.shape[0]):
        print(i, idx[i])
        if i < 3:
            is_efficient_bool[idx[i]] = True
        else:
            is_efficient_bool[idx[i]] = False    


    print(is_efficient_bool)

    return is_efficient_bool


    """
    Find the Pareto-efficient points.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep points not dominated
            is_efficient[i] = True  # Re-include this point
    return is_efficient


def create_empty_dataframe():
    dict = {
    "Model": [],
    "API": [],
    "Average latency": [],
    "MIoU": [],
    "quantized": []
    }

    df = pd.DataFrame(dict)

    return df

def interate_through_database(database, df):

    database = database[database["inference type"] == "seg"]

    #print(database)

    """
    models = [
        "lite-model_mobilenet_v2_100_224_fp32_1",
        "mobilenetv2-12",
        "mobilenet-v2-1.4-224_FP32",
        "mobilenet_v2",
        "lite-model_mobilenet_v2_100_224_uint8_1",
        "mobilenetv2-12-int8",
        "mobilenet_v2_q"
        ]
    
    quantized = [
        "lite-model_mobilenet_v2_100_224_uint8_1",
        "mobilenetv2-12-int8",
        "mobilenet_v2_q"
    ]
    """
    

    for i,r in database.iterrows():
        api = r["api"]
        mean_lat = r["latency avg"]
        miou = r["miou"]
        model_name = r["model_name"]

        if "int8" in model_name:
            q = "yes"
        else:
            q = "no"

        entry = {
            "Model": [model_name],
            "API": [api],
            "Average latency": [mean_lat],
            "MIoU": [miou],
            "quantized": [q]
        }
        df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 
    
    return df


if __name__ == "__main__":
    main()