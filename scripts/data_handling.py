
import plotly
import os
import pandas as pd
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def main():
    #print("Start")

    general_dir = os.path.dirname(os.path.realpath(__file__))
    general_dir = general_dir.split("scripts")[0]

    #print(general_dir)

    rasp_tf_dict, rasp_onnx_dict, rasp_ov_dict = get_raspberry_results(general_dir)
    ub_tf_dict, ub_onnx_dict, ub_ov_dict, ub_pyarmnn_dict = get_ubuntu_server_results(general_dir)

    for key in rasp_tf_dict:
        rasp_tf_dict.update({key:rasp_tf_dict[key]["inference time"]})

    for key in rasp_onnx_dict:
        rasp_onnx_dict.update({key:rasp_onnx_dict[key]["inference time"]})

    for key in rasp_ov_dict:
        rasp_ov_dict.update({key:rasp_ov_dict[key]["inference time"]})

    for key in ub_tf_dict:
        ub_tf_dict.update({key:ub_tf_dict[key]["inference time"]})

    for key in ub_onnx_dict:
        ub_onnx_dict.update({key:ub_onnx_dict[key]["inference time"]})

    for key in ub_ov_dict:
        ub_ov_dict.update({key:ub_ov_dict[key]["inference time"]})

    for key in ub_pyarmnn_dict:
        ub_pyarmnn_dict.update({key:ub_pyarmnn_dict[key]["inference time"]})

    print(rasp_ov_dict["mobilenet-v2"].mean())
    print(ub_ov_dict["mobilenet-v2"].mean())


    #comparing raspberry os and ubuntu server  
    fig = make_subplots(rows=1, cols=2)  

    fig.add_trace(
        go.Bar(x=["mobilenet_v3_large_fp32_tf", 
                  "mobilenet_v3_large_uint8_tf", 
                  "mobilenetv2-12_onnx", 
                  "mobilenetv2-12-int8_onnx",
                  "mobilenetv2_ov"
                  ],
               y=[rasp_tf_dict["lite-model_mobilenet_v3_large_100_224_fp32_1"].mean(), 
                  rasp_tf_dict["lite-model_mobilenet_v3_large_100_224_uint8_1"].mean(), 
                  rasp_onnx_dict["mobilenetv2-12"].mean(),
                  rasp_onnx_dict["mobilenetv2-12-int8"].mean(),
                  rasp_ov_dict["mobilenet-v2"].mean()
                  ],
                name = "raspberry os"),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=["mobilenet_v3_large_fp32_tf", 
                  "mobilenet_v3_large_uint8_tf", 
                  "mobilenetv2-12_onnx", 
                  "mobilenetv2-12-int8_onnx",
                  "mobilenetv2_ov"
                  ], 
                y=[ub_tf_dict["lite-model_mobilenet_v3_large_100_224_fp32_1"].mean(), 
                   ub_tf_dict["lite-model_mobilenet_v3_large_100_224_uint8_1"].mean(),
                   ub_onnx_dict["mobilenetv2-12"].mean(),
                   ub_onnx_dict["mobilenetv2-12-int8"].mean(),
                   ub_ov_dict["mobilenet-v2"].mean()
                   ],
                name = "ubuntu server"),
        row=1, col=1
    )



    fig.update_layout(height=600, width=800, title_text="Raspberry OS vs Ubuntu server")
    fig.show()

    fig = make_subplots(rows=1, cols=2)  

    fig.add_trace(
        go.Bar(x=["mobilenet_v3_large_fp32_tf", 
                  "mobilenet_v3_large_uint8_tf", 
                  "mobilenet_v3_small_fp32_tf", 
                  "mobilenet_v3_small_uint8_tf",
                  ],
               y=[ub_tf_dict["lite-model_mobilenet_v3_large_100_224_fp32_1"].mean(), 
                  ub_tf_dict["lite-model_mobilenet_v3_large_100_224_uint8_1"].mean(), 
                  ub_tf_dict["lite-model_mobilenet_v3_small_100_224_fp32_1"].mean(),
                  ub_tf_dict["lite-model_mobilenet_v3_small_100_224_uint8_1"].mean(),
                  ],
                  name = "tensorflow"),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=["mobilenet_v3_large_fp32_tf", 
                  "mobilenet_v3_large_uint8_tf", 
                  "mobilenet_v3_small_fp32_tf", 
                  "mobilenet_v3_small_uint8_tf",
                  ], 
                y=[ub_pyarmnn_dict["lite-model_mobilenet_v3_large_100_224_fp32_1"].mean(), 
                   ub_pyarmnn_dict["lite-model_mobilenet_v3_large_100_224_uint8_1"].mean(),
                   ub_pyarmnn_dict["lite-model_mobilenet_v3_small_100_224_fp32_1"].mean(),
                   ub_pyarmnn_dict["lite-model_mobilenet_v3_small_100_224_uint8_1"].mean()
                   ],
                   name = "pyarmnn"),
        row=1, col=1
    )

    fig.update_layout(height=600, width=800, title_text="Pyarmnnn vs Ubuntu Server")
    fig.show()


    print(search_file("lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", general_dir))
    print(os.path.getsize(search_file("lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", general_dir)))

    fig = px.scatter(y=[
            os.path.getsize(search_file("lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", general_dir)),
            os.path.getsize(search_file("lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", general_dir)),
            os.path.getsize(search_file("lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", general_dir)),
            os.path.getsize(search_file("lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", general_dir))
        ], 
        x=[
            ub_tf_dict["lite-model_mobilenet_v3_large_100_224_fp32_1"].mean(), 
            ub_tf_dict["lite-model_mobilenet_v3_large_100_224_uint8_1"].mean(),
            ub_tf_dict["lite-model_mobilenet_v3_small_100_224_fp32_1"].mean(),
            ub_tf_dict["lite-model_mobilenet_v3_small_100_224_uint8_1"].mean()
        ],
        color = ["mobilenet_v3_large_fp32", 
                  "mobilenet_v3_large_uint8", 
                  "mobilenet_v3_small_fp32", 
                  "mobilenet_v3_small_uint8",
                  ],
        text = [
            "large_fp32",
            "large_int8",
            "small_fp32",
            "small_int8"
        ],
        labels = {
            "x": "latency",
            "y": "size of model",
            "color": ".tflite models",
        },
        title = "Model inference in tensorflow size and latency examination"
        )
    fig.update_traces(textposition="bottom right")
    fig.show()


    #fig = go.Figure()

    # import packages
    #import numpy as np
    # create dummy data
    #vals = np.ceil(100 * np.random.rand(5)).astype(int)
    #keys = ["A", "B", "C", "D", "E"]

    #print(rasp_tf_dict["lite-model_mobilenet_v3_large_100_224_fp32_1"].mean())
    #print(ub_tf_dict["lite-model_mobilenet_v3_large_100_224_fp32_1"].mean())



    # plot data
    #fig.add_trace(
    #go.Bar(x=["rasp","ub"],y=[rasp_tf_dict["lite-model_mobilenet_v3_large_100_224_fp32_1"].mean(), ub_tf_dict["lite-model_mobilenet_v3_large_100_224_fp32_1"].mean()])
    #)
    #fig.update_layout(height=600, width=600)
    #fig.show()

    #fig = make_subplots(rows=1, cols=2)

    #fig.add_trace(
    #    go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
    #    row=1, col=1
    #)

    #fig.add_trace(
    #    go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
    #    row=1, col=2
    #)

    #fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
    #fig.show()


    #for key in rasp_ov_dict:
    #    print(rasp_ov_dict[key])
    #    print(rasp_ov_dict[key]["inference time"])

        


def get_raspberry_results(general_dir): 
    rasp_result_path = os.path.join(general_dir, "results_raspberry_os", "class")

    tf_dict = {}

    #tf
    rasp_result_path_tf = os.path.join(rasp_result_path, "tf")
    df = pd.read_csv(os.path.join(rasp_result_path_tf, "lite-model_mobilenet_v3_large_100_224_fp32_1", "output", "2023_10_10_18_14.csv"))
    tf_dict.update({"lite-model_mobilenet_v3_large_100_224_fp32_1": df})
    df = pd.read_csv(os.path.join(rasp_result_path_tf, "lite-model_mobilenet_v2_100_224_fp32_1", "output", "2023_10_10_18_19.csv"))
    tf_dict.update({"lite-model_mobilenet_v2_100_224_fp32_1": df})
    df = pd.read_csv(os.path.join(rasp_result_path_tf, "lite-model_mobilenet_v2_100_224_uint8_1", "output", "2023_10_10_18_23.csv"))
    tf_dict.update({"lite-model_mobilenet_v2_100_224_uint8_1": df})
    df = pd.read_csv(os.path.join(rasp_result_path_tf, "lite-model_mobilenet_v3_large_100_224_uint8_1", "output", "2023_10_10_18_15.csv"))
    tf_dict.update({"lite-model_mobilenet_v3_large_100_224_uint8_1": df})
    df = pd.read_csv(os.path.join(rasp_result_path_tf, "lite-model_mobilenet_v3_small_100_224_fp32_1", "output", "2023_10_10_18_16.csv"))
    tf_dict.update({"lite-model_mobilenet_v3_small_100_224_fp32_1": df})
    df = pd.read_csv(os.path.join(rasp_result_path_tf, "lite-model_mobilenet_v3_small_100_224_uint8_1", "output", "2023_10_10_18_19.csv"))
    tf_dict.update({"lite-model_mobilenet_v3_small_100_224_uint8_1": df})

    onnx_dict = {}

    rasp_result_path_onnx = os.path.join(rasp_result_path, "onnx")
    #print(os.path.exists(os.path.join(rasp_result_path_onnx, "mobilenetv2-12", "output")))
    #sys.exit()
    df = pd.read_csv(os.path.join(rasp_result_path_onnx, "mobilenetv2-12", "output", "2023_10_10_18_25.csv"))
    onnx_dict.update({"mobilenetv2-12":df})
    df = pd.read_csv(os.path.join(rasp_result_path_onnx, "mobilenetv2-7", "output", "2023_10_10_18_29.csv"))
    onnx_dict.update({"mobilenetv2-7":df})
    df = pd.read_csv(os.path.join(rasp_result_path_onnx, "mobilenetv2-10", "output", "2023_10_10_18_30.csv"))
    onnx_dict.update({"mobilenetv2-10":df})
    df = pd.read_csv(os.path.join(rasp_result_path_onnx, "mobilenetv2-12-int8", "output", "2023_10_10_18_31.csv"))
    onnx_dict.update({"mobilenetv2-12-int8":df})

    ov_dict = {}

    rasp_result_path_ov = os.path.join(rasp_result_path, "ov")
    df = pd.read_csv(os.path.join(rasp_result_path_ov, "mobilenet-v2", "output", "2023_10_10_18_30.csv"))
    ov_dict.update({"mobilenet-v2":df})

    return tf_dict, onnx_dict, ov_dict




def get_ubuntu_server_results(general_dir):
    ub_server_result_path = os.path.join(general_dir, "results_ubuntu_server", "class")

    tf_dict = {}

    ub_server_result_path_tf = os.path.join(ub_server_result_path, "tf")
    df = pd.read_csv(os.path.join(ub_server_result_path_tf, "lite-model_mobilenet_v3_large_100_224_fp32_1", "output", "2023_10_11_15_32.csv"))
    tf_dict.update({"lite-model_mobilenet_v3_large_100_224_fp32_1": df})
    df = pd.read_csv(os.path.join(ub_server_result_path_tf, "lite-model_mobilenet_v2_100_224_fp32_1", "output", "2023_10_11_15_37.csv"))
    tf_dict.update({"lite-model_mobilenet_v2_100_224_fp32_1": df})
    df = pd.read_csv(os.path.join(ub_server_result_path_tf, "lite-model_mobilenet_v2_100_224_uint8_1", "output", "2023_10_11_15_38.csv"))
    tf_dict.update({"lite-model_mobilenet_v2_100_224_uint8_1": df})
    df = pd.read_csv(os.path.join(ub_server_result_path_tf, "lite-model_mobilenet_v3_large_100_224_uint8_1", "output", "2023_10_11_15_32.csv"))
    tf_dict.update({"lite-model_mobilenet_v3_large_100_224_uint8_1": df})
    df = pd.read_csv(os.path.join(ub_server_result_path_tf, "lite-model_mobilenet_v3_small_100_224_fp32_1", "output", "2023_10_11_15_33.csv"))
    tf_dict.update({"lite-model_mobilenet_v3_small_100_224_fp32_1": df})
    df = pd.read_csv(os.path.join(ub_server_result_path_tf, "lite-model_mobilenet_v3_small_100_224_uint8_1", "output", "2023_10_11_15_35.csv"))
    tf_dict.update({"lite-model_mobilenet_v3_small_100_224_uint8_1": df})

    onnx_dict = {}

    ub_server_result_path_onnx = os.path.join(ub_server_result_path, "onnx")
    df = pd.read_csv(os.path.join(ub_server_result_path_onnx, "mobilenetv2-12", "output", "2023_10_11_15_40.csv"))
    onnx_dict.update({"mobilenetv2-12":df})
    df = pd.read_csv(os.path.join(ub_server_result_path_onnx, "mobilenetv2-7", "output", "2023_10_11_15_41.csv"))
    onnx_dict.update({"mobilenetv2-7":df})
    df = pd.read_csv(os.path.join(ub_server_result_path_onnx, "mobilenetv2-10", "output", "2023_10_11_15_42.csv"))
    onnx_dict.update({"mobilenetv2-10":df})
    df = pd.read_csv(os.path.join(ub_server_result_path_onnx, "mobilenetv2-12-int8", "output", "2023_10_11_15_41.csv"))
    onnx_dict.update({"mobilenetv2-12-int8":df})

    ov_dict = {}

    ub_server_result_path_ov = os.path.join(ub_server_result_path, "ov")
    df = pd.read_csv(os.path.join(ub_server_result_path_ov, "mobilenet-v2", "output", "2023_10_11_15_45.csv"))
    ov_dict.update({"mobilenet-v2": df})

    pyarmnn_dict = {}

    ub_server_result_path_pyarmnn = os.path.join(ub_server_result_path, "pyarmnn")
    df = pd.read_csv(os.path.join(ub_server_result_path_pyarmnn, "lite-model_mobilenet_v3_large_100_224_fp32_1", "output", "2023_10_11_15_38.csv"))
    pyarmnn_dict.update({"lite-model_mobilenet_v3_large_100_224_fp32_1": df})
    df = pd.read_csv(os.path.join(ub_server_result_path_pyarmnn, "lite-model_mobilenet_v2_100_224_fp32_1", "output", "2023_10_11_15_39.csv"))
    pyarmnn_dict.update({"lite-model_mobilenet_v2_100_224_fp32_1": df})
    df = pd.read_csv(os.path.join(ub_server_result_path_pyarmnn, "lite-model_mobilenet_v2_100_224_uint8_1", "output", "2023_10_11_15_40.csv"))
    pyarmnn_dict.update({"lite-model_mobilenet_v2_100_224_uint8_1": df})
    df = pd.read_csv(os.path.join(ub_server_result_path_pyarmnn, "lite-model_mobilenet_v3_large_100_224_uint8_1", "output", "2023_10_11_15_39.csv"))
    pyarmnn_dict.update({"lite-model_mobilenet_v3_large_100_224_uint8_1": df})
    df = pd.read_csv(os.path.join(ub_server_result_path_pyarmnn, "lite-model_mobilenet_v3_small_100_224_fp32_1", "output", "2023_10_11_15_39.csv"))
    pyarmnn_dict.update({"lite-model_mobilenet_v3_small_100_224_fp32_1": df})
    df = pd.read_csv(os.path.join(ub_server_result_path_pyarmnn, "lite-model_mobilenet_v3_small_100_224_uint8_1", "output", "2023_10_11_15_39.csv"))
    pyarmnn_dict.update({"lite-model_mobilenet_v3_small_100_224_uint8_1": df})

    return tf_dict, onnx_dict, ov_dict, pyarmnn_dict

    #tf

    #pyarmnn

    #onnx

    #ov

def search_file(file_name, directory):

    for dirpath, dirnames, files in os.walk(directory):

        for file in files:
            if file == file_name:
                return os.path.join(dirpath, file_name)

    sys.exit(f"Error: The file {file_name} does not exist in the directory {directory}!")


if __name__ == "__main__":
    main()
