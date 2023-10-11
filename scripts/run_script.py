import subprocess
import argparse
import sys

def main():
    #dic = create_dictionary()
    handle_arguments()
    

def handle_arguments():
    number = int(sys.argv[1])

    #classification 1, 00 tf, 20 pyarmnn, 40 onnx, 60 pytorch, 80 openvino
    if number == 100:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1"], shell=False)
    if number == 101:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1",  "-so"], shell=False)
    if number == 102:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 103:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 104:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 105:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 106:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 107:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 120:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class",  "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1"], shell=False)
    if number == 121:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class",  "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so"], shell=False)
    if number == 122:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class",  "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so", "-ri", "50"], shell=False)
    if number == 123:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 124:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 125:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 126:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 127:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 140:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1"], shell=False)
    if number == 141:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so"], shell=False)
    if number == 142:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 143:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12-int8.onnx", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 144:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-7.onnx", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 145:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-10.onnx", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)
    if number == 160:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1"], shell=False)
    if number == 161:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so"], shell=False)
    if number == 162:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so","-ri", "50"], shell=False)
    if number == 163:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so","-ri", "50"], shell=False)
    if number == 180:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2.xml", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1"], shell=False)
    if number == 181:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2.xml", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so"], shell=False)
    if number == 182:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2.xml", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so","-ri", "50"], shell=False)




if __name__ == "__main__":
    main()
