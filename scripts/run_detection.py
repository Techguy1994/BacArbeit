import subprocess
import argparse
import sys

def main():
    handle_arguments()
    

def handle_arguments():
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module for Detection')

        #basic settings
    parser.add_argument("-n", '--n', help='inference API: pyarmnn, tf, ov, onnx, pytorch', required=True) #which api to use
    args = parser.parse_args()
    number = int(args.n)

    """
    number code:
     first digit: OS: Ubuntu --> 1, RaspOS --> 5
     second digit: thread count count: 1 thread --> 1, ...
     third digit: inf type: classificaiton --> 1, object detection --> 2, image segmenation --> 3
     fourht digit: FW: delegate --> 0, tflite-runtime --> 2, onnx --> 4, openvino --> 6, pytorch --> 8
     fifth digit: run type: latency --> 0, accuracy --> 5
     sixth digit: model_number --> custom

    models: 
            tflite:
                lite-model_mobilenet_v3_large_100_224_fp32_1.tflite:    00
                lite-model_mobilenet_v3_large_100_224_uint8_1.tflite:   01
                lite-model_mobilenet_v3_small_100_224_fp32_1.tflite:    02
                lite-model_mobilenet_v3_small_100_224_uint8_1.tflite:   03
                lite-model_mobilenet_v2_100_224_fp32_1.tflite:          04
                lite-model_mobilenet_v2_100_224_uint8_1.tflite:         05
            
            onnx: 
                mobilenetv3_large_100_Opset17.onnx: 50
                mobilenetv3_small_075_Opset17.onnx: 51
                mobilenetv3_small_050_Opset17.onnx: 52
                mobilenetv2-12.onnx:                53
                mobilenetv2-12-int8.onnx:           54

            openvino:
                mobilenet-v3-large-1.0-224-tf_FP32.xml: 100
                mobilenet-v3-large-1.0-224-tf_FP32.xml: 101
                mobilenet-v2-1.4-224_FP32.xml:          102

            pytorch:
                mobilenet_v3_large:     150
                mobilenet_v3_large_q:   151
                mobilenet_v3_small:     152
                mobilenet_v2:           153
                mobilenet_v2_q:         154
    """

if __name__ == "__main__":
    main()