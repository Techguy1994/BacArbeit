import pandas as pd 
import sys

def main():
    df = pd.read_csv("database.csv")

    df = df[df["os"].str.contains("raspberryos")==False]
    df = df[df["model_name"].str.contains("mobilenetv2-12-int8")==False]
    df = df[df["model_name"].str.contains("mobilenetv3_small_075_Opset17")==False]
    df = df[df["model_name"].str.contains("mobilenet_v3_large_q")==False]
    df = df[df["model_name"].str.contains("mobilenet_v2_q")==False]

    df['api'] = df['api'].replace('onnx', 'ONNX')
    df['api'] = df['api'].replace('ov', 'OpenVINO')
    df['api'] = df['api'].replace('tf', 'TFLite')
    df['api'] = df['api'].replace('delegate', 'Arm NN Delegate')
    df['api'] = df['api'].replace('pytorch', 'PyTorch')

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

    df['model_name'] = df['model_name'].replace('yolov5l','Yolov5l')
    df['model_name'] = df['model_name'].replace('yolov5m','Yolov5m')
    df['model_name'] = df['model_name'].replace('yolov5n','Yolov5n')
    df['model_name'] = df['model_name'].replace('yolov5s','Yolov5s')

    df = df.drop('os', axis=1)

    df = df.rename(columns={
        "api": "Frameworks",
        "model_name": "Model",
        "latency avg": "Mean Latency [s]",
        "top1": "Top1",
        "top5": "Top5",
        "map": "mAP",
        "miou": "mIoU"
    })

    df["Median Latency [s]"] = 0.0
    df["Standard Deviation [s]"] = 0.0 
    df["Variance [s²]"] = 0.0

    for i,r in df.iterrows():
        latency_link = r["latency"]

        #print(model_name, threads, api, latency_link)

        lat = pd.read_csv(latency_link)

        lat.to_csv("test.csv")

        median = lat["inference time"].median()
        var = lat["inference time"].var()
        std = lat["inference time"].std()

        df.at[i, "Median Latency [s]"] = median
        df.at[i, "Variance [s²]"] = var
        df.at[i, "Standard Deviation [s]"] = std

    df.to_csv("database_updated.csv")

if __name__ == "__main__":
    main()