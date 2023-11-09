import subprocess
import argparse
import sys

def main():
    #dic = create_dictionary()
    handle_arguments()
    

def handle_arguments():
    number = int(sys.argv[1])
    print(number)

    #classification 1, 00 tf, 20 pyarmnn, 40 onnx, 60 pytorch, 80 openvino
    if number == 100:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
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
    if number == 200:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)

    # object detection
    if number == 400:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so", "-ri", "10"], shell=False)
    if number == 401:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so", "-ri", "10"], shell=False)
    if number == 402:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so", "-ri", "10"], shell=False)
    if number == 403:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so", "-ri", "10"], shell=False)
    if number == 404:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so", "-ri", "10"], shell=False)
    if number == 405:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so", "-ri", "10"], shell=False)
    if number == 410:
        #0.25 confidence thresh for yolo v5 as per detect.py in the github page yolov5 
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgd", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 411:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgd", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 412:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgd", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 413:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgd", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 414:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgd", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 415:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgd", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)


    # segmentation
    if number == 800:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv3-cityscapes_1_default_2.tflite", "-imgd", "-l", "imagenet_classes.txt", "-c", "cityscapes","-opd", "-ni", "1", "-so", "-ri", "10"], shell=False)
    if number == 801:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv3-cityscapes_1_default_2.tflite", "-imgd", "-l", "imagenet_classes.txt", "-c", "cityscapes","-opd", "-ni", "1", "-so", "-ri", "10"], shell=False)
    if number == 802:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv3-cityscapes_1_default_2.tflite", "-imgd", "-l", "imagenet_classes.txt", "-c", "cityscapes","-opd", "-ni", "1", "-so", "-ri", "10"], shell=False)
    if number == 803:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv3-cityscapes_1_default_2.onnx", "-imgd", "-l", "imagenet_classes.txt", "-c", "cityscapes","-opd", "-ni", "1", "-so", "-ri", "10"], shell=False)
    if number == 804:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_large", "-imgd", "-l", "imagenet_classes.txt", "-c", "cityscapes","-opd", "-ni", "1", "-so", "-ri", "10"], shell=False)

if __name__ == "__main__":
    main()
