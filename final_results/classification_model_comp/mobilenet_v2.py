import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import sys
import matplotlib.pyplot as plt

def main():
    database = pd.read_csv("database.csv")
    csv = "mobilenet_v2_scatter.csv"

    df = create_empty_dataframe()

    df = interate_through_database(database, df)

    df = df.sort_values(by=['mean_lat', 'top1'])

    pareto_points_top1 = pareto_top1(df)
    pareto_points_top5 = pareto_top5(df)
    print(pareto_points_top1)


    # Extract latency and accuracy columns
    latency = df['mean_lat'].values
    accuracy = df['top1'].values

    # Combine into a single array for processing
    scatter_data = np.column_stack((accuracy, latency))

    # Sort by accuracy (descending), then by latency (ascending)
    sorted_data = scatter_data[np.lexsort((scatter_data[:, 1], -scatter_data[:, 0]))]

    # Compute Pareto front
    pareto_front = []
    lowest_latency = np.inf

    for acc, lat in sorted_data:
        if lat < lowest_latency:
            pareto_front.append((acc, lat))
            lowest_latency = lat

    pareto_front = np.array(pareto_front)

    # Plot the scatter plot and Pareto front
    plt.figure(figsize=(10, 6))
    plt.scatter(scatter_data[:, 0], scatter_data[:, 1], label='All Points', color='blue', alpha=0.6)
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], label='Pareto Front', color='red', marker='o', linestyle='-')
    plt.xlabel('Top-1 Accuracy', fontsize=12)
    plt.ylabel('Mean Latency (s)', fontsize=12)
    plt.title('Pareto Front: Accuracy vs Latency', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

    sys.exit()


    print(df)
    df.to_csv(csv)

    #top1 plot
    fig = px.scatter(df, x="top1", y="mean_lat", color="api", symbol="model_name", title="MobilenNet V2")
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

    #top5 plot 

    fig = px.scatter(df, x="top5", y="mean_lat", color="api", symbol="model_name", title="MobilenNet V2")
    fig.update_traces(marker=dict(size=20),
                    selector=dict(mode='markers'))
    #fig.add_trace(
    #        go.Scatter(
    #            x=pareto_points_top5['top5'],
    #            y=pareto_points_top5['mean_lat'],
    #            mode='lines+markers',
    #            name='Pareto Front',
    #            line=dict(color='red', width=2),
     #           marker=dict(size=8)
    #    )
    #)
    fig.show()




def pareto_top1(df):
    print("Df")
    print(df)
    
    pareto_df = df[['mean_lat', 'top1']]

    print("just lat and top1")
    print(pareto_df)


    pareto_indices = is_pareto_efficient(pareto_df.to_numpy())
    pareto_points = pareto_df[pareto_indices]

    return pareto_points

def pareto_top5(df):
    print("Df")
    print(df)
    
    pareto_df = df[['mean_lat', 'top5']]

    print("just lat and top5")
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
    "model_name": [],
    "api": [],
    "mean_lat": [],
    "top1": [],
    "top5": [],
    "quantized": []
    }

    df = pd.DataFrame(dict)

    return df

def interate_through_database(database, df):



    database = database[(database["os"] == "ubuntus")]

    

    print("----")
    print(database)
    print("---")


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

    

    for i,r in database.iterrows():
        if r["model_name"] in models:
            print(r["model_name"])
            api = r["api"]
            mean_lat = r["latency avg"]
            top1 = r["top1"]
            top5 = r["top5"]

            if r["model_name"] in quantized:
                q = "yes"
            else:
                q = "no"

            entry = {
                "model_name": [r["model_name"]],
                "api": [api],
                "mean_lat": [mean_lat],
                "top1": [top1],
                "top5": [top5],
                "quantized": [q]
            }
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) 
    
    return df


if __name__ == "__main__":
    main()