import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

def main():
    df = pd.read_csv("database.csv")

    df = df[df["os"].str.contains("raspberryos")==False]
    df = df[df["model_name"].str.contains("_q")==False]
    df = df[df["model_name"].str.contains("int8")==False]

    df = df.drop('top1', axis=1)
    df = df.drop('top5', axis=1)
    df = df.drop('map', axis=1)
    df = df.drop('miou', axis=1)
    df = df.drop('os', axis=1)

    df['api'] = df['api'].replace('onnx', 'Onnx')
    df['api'] = df['api'].replace('ov', 'Openvino')
    df['api'] = df['api'].replace('tf', 'Tensorflow')
    df['api'] = df['api'].replace('delegate', 'Armnn Delegate')
    df['api'] = df['api'].replace('pytorch', 'PyTorch')

    df["variance"] = 0

    for i,r in df.iterrows():



        model = r["model_name"]
        latency_link = r["latency"]
        avg = r["latency avg"]

        lat = pd.read_csv(latency_link)


        #print(lat["inference time"].var())
        #print(lat["inference time"].mean(), avg)
        #print(i,r)

        df.at[i, "variance"]= lat["inference time"].var()

    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v2_100_224_fp32_1', 'MobileNetV2')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v2_100_224_uint8_1', 'MobileNetV2 INT8')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v3_large_100_224_fp32_1', 'MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v3_large_100_224_uint8_1', 'MobileNetV3 Large INT8')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v3_small_100_224_fp32_1', 'MobileNetV3 Small')
    df['model_name'] = df['model_name'].replace('lite-model_mobilenet_v3_small_100_224_uint8_1', 'MobileNetV3 Small INT8')

    df['model_name'] = df['model_name'].replace('mobilenetv2-12', 'MobileNetV2')
    df['model_name'] = df['model_name'].replace('mobilenetv3_large_100_Opset17', 'MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('mobilenetv3_small_050_Opset17', 'MobileNetV3 Small')

    df['model_name'] = df['model_name'].replace('mobilenet_v2', 'MobileNetV2')
    df['model_name'] = df['model_name'].replace('mobilenet_v3_large', 'MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('mobilenet_v3_small', 'MobileNetV3 Small')

    df['model_name'] = df['model_name'].replace('mobilenet-v2-1.4-224_FP32', 'MobileNetV2')
    df['model_name'] = df['model_name'].replace('mobilenet-v3-large-1.0-224-tf_FP32', 'MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('mobilenet-v3-small-1.0-224-tf_FP32', 'MobileNetV3 Small')

    df['model_name'] = df['model_name'].replace('deeplabv3_FP32', 'DeeplabV3 MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('deeplabv3_mobilenet_v3_large', 'DeeplabV3 MobileNetV3 Large')
    df['model_name'] = df['model_name'].replace('lite-model_deeplabv3-mobilenetv2_1_default_1', 'DeeplabV3 MobileNetV2 FP32')
    df['model_name'] = df['model_name'].replace('lite-model_deeplabv3-mobilenetv2-int8_1_default_1', 'DeeplabV3 MobileNetV2 INT8')
        

    df.to_csv("temp.csv")







if __name__ == "__main__":
    main()