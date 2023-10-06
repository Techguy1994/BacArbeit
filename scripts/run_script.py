import subprocess
import argparse
import sys

def main():
    #dic = create_dictionary()
    handle_arguments()
    print("Hello")
    

def handle_arguments():
    number = int(sys.argv[1])

    if number == 100:
        subprocess.call("python3 run_inference.py -api tf -t class -m lite-model_mobilenet_v3_large_100_224_fp32_1.tflite -imgd -l imagenet_classes.txt -opd -ni 1", shell=True)
    if number == 120:
        subprocess.call("python3 run_inference.py -api pyarmnn -t class -m lite-model_mobilenet_v3_large_100_224_fp32_1.tflite -imgd -l imagenet_classes.txt -opd -ni 1", shell=True)
    if number == 140:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1"], shell=False)
    if number == 141:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so"], shell=False)
    if number == 142:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so", "-ri", "50"], shell=False)


if __name__ == "__main__":
    main()
