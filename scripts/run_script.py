import subprocess
import argparse
import sys

def main():
    handle_arguments()
    

def handle_arguments():
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module for Classification')

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
    #raspberry os
    # latency
    #classification, tflite
    if number == 5112000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "1"], shell=False)
    if number == 5312000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5312001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5312002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)
    
    if number == 5312003:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5312004:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5312005:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    #classification, delegate 
    if number == 5110000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "1"], shell=False)
    if number == 5310000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5310001:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5310002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5310003:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5310004:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5310005:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)
    
    # classification, onnx
    if number == 5114050:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "1"], shell=False)
    if number == 5314050:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5314051:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_075_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5314052:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_050_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5314053:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 5314054:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12-int8.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)


    #classification, openvino
    if number == 51160100:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "1"], shell=False)
    if number == 53160100:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 53160101:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 53160102:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    #classificaiton, pytorch

    if number == 53180150:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 53180151:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 53180152:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_small", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 53180153:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    if number == 53180154:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberryos", "-num_thr", "3"], shell=False)

    # ubuntu server 
    # classification accuracy runs
    if number == 13185150:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 13185151:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntus", "-num_thr", "3"], shell=False)

    #classification, openvino 
    if number == 13160100:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 13160101:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 13160102:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)


    if number == 13165100:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 13165101:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000//", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 13165102:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntus", "-num_thr", "3"], shell=False)

    #tf latency runs
    if number == 1312000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1312001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1312002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    if number == 1312003:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1312004:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1312005:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #pytorch lat runs
    if number == 13180150:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 13180151:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 13180152:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_small", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 13180153:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 13180154:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    
    #delegate latency runs 
    if number == 1110000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1310000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310001:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310003:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310004:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310005:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    if number == 13120001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "10", "-so", "-os", "ubuntus", "-num_thr", "3", "-p", "cprofiler"], shell=False)





    sys.exit()
    #test runs
    #classification 1, 00 tf, 20 pyarmnn, 40 onnx, 60 pytorch, 80 openvino, pyarmnn outdated replaced with delegate test run on 190
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
    if number == 110:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_val/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 111:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_test_images/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)

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
    if number == 150:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_val/",  "-l", "imagenet_classes.txt", "-opd", "-ni", "1"], shell=False)
    if number == 160:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1"], shell=False)
    if number == 161:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so"], shell=False)
    if number == 162:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so","-ri", "50"], shell=False)
    if number == 163:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so","-ri", "50"], shell=False)

    #thread latency testing
    if number == 170:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "1"], shell=False)
    if number == 171:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "2"], shell=False)
    if number == 172:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 173:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)

    if number == 174:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "1"], shell=False)
    if number == 175:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "2"], shell=False)
    if number == 176:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 177:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
        
    if number == 178:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "1"], shell=False)
    if number == 179:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "2"], shell=False)
    if number == 180:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 181:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    
    if number == 182:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "1"], shell=False)
    if number == 183:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "2"], shell=False)
    if number == 184:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 185:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    
    
    if number == 186:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "1"], shell=False)
    if number == 187:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "2"], shell=False)
    if number == 188:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 189:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    
    if number == 190:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "1"], shell=False)
    if number == 191:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "2"], shell=False)
    if number == 192:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 193:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
 
    #inference classification accuracy runs
    #tflite
    if number == 200:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 201:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 202:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 203:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 204:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 205:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)

    #delegate
    if number == 206:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 207:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 208:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 209:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 210:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 211:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    #onnx
    if number == 212:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 213:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12-int8.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 214:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 215:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_075_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 216:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_050_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    #pytorch
    if number == 217:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 218:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_small", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 219:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 220:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 221:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    #openvino
    if number == 222:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 223:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 224:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000//", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 225:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 226:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 227:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)

    #latency testing on ubuntu server
    #tflite and pyarmnn inference classification
    #tflite
    if number == 228:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 229:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 230:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 231:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 232:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 233:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)

    #tflite delegate
    if number == 234:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 235:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 236:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 237:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 238:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 239:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    #onnx
    if number == 240:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 241:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12-int8.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 242:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 243:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_075_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 244:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_050_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    #pytroch
    if number == 245:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 246:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_small", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 247:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 248:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 249:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    #openvino
    if number == 290:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 291:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 292:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)

    #latency testing on raspberry os
    #tflite
    if number == 1228:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1229:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1230:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1231:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1232:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1233:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)

    #tflite delegate
    if number == 1234:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1235:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1236:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1237:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1238:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1239:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    #onnx
    if number == 1240:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1241:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12-int8.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1242:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1243:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_075_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1244:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_050_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    #pytroch
    if number == 1245:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1246:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_small", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1247:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1248:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1249:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    #openvino
    if number == 1290:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1291:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os", "-num_thr", "3"], shell=False)
    if number == 1292:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "raspberry_os32", "-num_thr", "3"], shell=False)

    # comparing pyarmnn and delegate latency 
    if number == 250:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "10", "-so", "-num_thr", "3"], shell=False)
    if number == 251:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "10", "-so", "-num_thr", "3"], shell=False)
    #comparision tf and delegate when using different number of threads for mobilenetv3 large
    if number == 252:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "10", "-num_thr", "1"], shell=False)
    if number == 253:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "10", "-num_thr", "2"], shell=False)
    if number == 254:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "10", "-num_thr", "3"], shell=False)
    if number == 255:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "10", "-num_thr", "4"], shell=False)
    if number == 256:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "10", "-num_thr", "1"], shell=False)
    if number == 257:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "10", "-num_thr", "2"], shell=False)
    if number == 258:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "10", "-num_thr", "3"], shell=False)
    if number == 259:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "10", "-num_thr", "4"], shell=False)
    #comparision tf and delegate when using different number of threads for mobilenetv2 large
    if number == 260:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "1"], shell=False)
    if number == 261:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "2"], shell=False)
    if number == 262:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "3"], shell=False)
    if number == 263:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "4"], shell=False)
    if number == 264:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "1"], shell=False)
    if number == 265:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "2"], shell=False)
    if number == 266:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "3"], shell=False)
    if number == 267:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "4"], shell=False)

    #tflite latnecy tests
    if number == 270:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s"], shell=False)
    
    #onnx latency tests
    if number == 271:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgd", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "25", "-so", "-num_thr", "3"], shell=False)
    #tf latency test
    if number == 272:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "25", "-num_thr", "3"], shell=False)
    # openvino latency test
    if number == 273:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP16.xml", "-imgd", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "25", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    #pytorch latency testing
    if number == 274:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgd", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "25", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 275:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgd", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "25", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 276:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2_q", "-imgd", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "25", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    
    #openvin comparing FP16 and FP32
    if number == 280:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-os", "ubuntu_s"], shell=False)
    if number == 281:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-os", "ubuntu_s"], shell=False)
    if number == 282:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-os", "ubuntu_s"], shell=False)
    if number == 283:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-os", "ubuntu_s"], shell=False)
    if number == 284:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgd", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-os", "ubuntu_s"], shell=False)
    if number == 285:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP16.xml", "-imgd", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-os", "ubuntu_s"], shell=False)

    

    
    #mac os test runs for classification 
    if number == 300:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_val/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 301:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_val/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 340:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_val/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 360:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_val/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 380:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 381:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP16.xml", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 382:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP32.xml", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 383:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP16.xml", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    
    if number == 390:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_test_images/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 391:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_test_images/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 392:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_test_images/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 393:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_test_images/", "-l", "imagenet_classes.txt", "-opd", "-ni", "1"], shell=False)
    if number == 394:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP32.xml", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 395:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP32.xml", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_test_images/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 396:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP32.xml", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_test_images/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "100", "-so"], shell=False)

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
    
    if number == 420:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 421:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 422:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 423:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 424:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)


    if number == 430:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/coco_test_images/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 433:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/coco_test_images/",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)



    if number == 440:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/Users/marounel-chayeb/BacArbeit/coco_test_images/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 442:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/Users/marounel-chayeb/BacArbeit/coco_test_images/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 443:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/Users/marounel-chayeb/BacArbeit/coco_test_images",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 444:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/Users/marounel-chayeb/BacArbeit/coco_test_images/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)

    if number == 450:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 452:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 453:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 454:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)

    #test pytorch loading of model
    if number == 490:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "10", "-os", "macos"], shell=False)
    if number == 491:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov3.pt", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "10", "-os", "macos"], shell=False)
    if number == 492:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov3-tiny.pt", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "10", "-os", "macos"], shell=False)
    if number == 493:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8l.pt", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "10", "-os", "macos"], shell=False)
    if number == 494:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8m.pt", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "10", "-os", "macos"], shell=False)
    if number == 495:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8n.pt", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "10", "-os", "macos"], shell=False)
    if number == 496:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8s.pt", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "10", "-os", "macos"], shell=False)
    if number == 497:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov7.pt", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "10", "-os", "macos"], shell=False)
    if number == 498:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov7-tiny.pt", "-imgp", "/Users/marounel-chayeb/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "10", "-os", "macos"], shell=False)
    
    

    # run the accuracy inference 
    if number == 500:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 501:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 502:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 503:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5s-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)

    if number == 520:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 521:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 522:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 523:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)

    if number == 540:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 541:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 542:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 543:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)

    if number == 550:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 551:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5m", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 552:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)
    if number == 553:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5s", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25"], shell=False)

    # run the latency numbers
    #yolov5
    if number == 600:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 601:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 602:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 603:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5s-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 604:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 605:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 606:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 607:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 608:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 609:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 610:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 611:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 612:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 613:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5m", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 614:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 615:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5s", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    # yolov3
    if number == 616:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov3-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 617:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov3-tiny_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 618:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov3.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 619:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov3-tiny.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 620:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov3.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 621:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov3-tiny.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 622:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov3", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 623:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov3-tiny", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    #622 amd 623 pytorch
    #yolov7
    if number == 624:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov7-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 625:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov7-tiny-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 626:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov3.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 627:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov3-tiny.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 628:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov7.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 629:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov7-tiny.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    #630 and 631 pytorch
    #yolov8
    if number == 632:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8l_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 633:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8m_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 634:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8n_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 635:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8s_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 636:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 637:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 638:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 639:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 640:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 641:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 642:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 643:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 644:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 615:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8m", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 616:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 617:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8s", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
        

    # segmentation
    # test runs
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

    if number == 810:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv3-cityscapes_1_default_2.tflite", "-imgd", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 811:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2-int8_1_default_1.tflite", "-imgd", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 812:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgd", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 813:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgd", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 814:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.onnx", "-imgd", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 815:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "seg", "-m", "deeplabv3.xml", "-imgd", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 816:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgd", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 820:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_large", "-imgd", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 821:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_small", "-imgd", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    
    # accuracy runs
    if number == 850:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 851:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 852:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2-int8_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 853:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2-int8_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 854:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/",  "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 855:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "seg", "-m", "deeplabv3_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 856:
         subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 857:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "seg", "-m", "deeplabv3_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)

    if number == 890:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_large.onnx", "-imgp", "/Users/marounel-chayeb/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "macos"], shell=False)
    if number == 891:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.onnx", "-imgp", "/Users/marounel-chayeb/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)

    # deeplabv3 acurracy tests tflite,onnx converted
    if number == 900:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_large.onnx", "-imgp", "/Users/marounel-chayeb/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "macos"], shell=False)
    if number == 901:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_large.tflite", "-imgp", "/Users/marounel-chayeb/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 902:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "deeplabv3_257_mv_gpu.tflite", "-imgp", "/Users/marounel-chayeb/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    if number == 903:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "1.tflite", "-imgp", "/Users/marounel-chayeb/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)

    #deeplabv3+ tests

    if number == 910:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "seg", "-m", "best_deeplabv3plus_mobilenet_voc_os16.pth", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/",  "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)

    # classification 3 vs 4 thread comparison runs
    # 3 thread
    #onnx
    if number == 3240:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 3241:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12-int8.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 3242:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 3243:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_075_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 3244:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_050_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    #pytroch
    if number == 3245:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 3246:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_small", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 3247:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 3248:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 3249:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    #openvino
    if number == 3290:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 3291:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)
    if number == 3292:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "3"], shell=False)

    # 4 thread
    #onnx
    if number == 4240:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    if number == 4241:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12-int8.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    if number == 4242:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    if number == 4243:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_075_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    if number == 4244:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_050_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    #pytroch
    if number == 4245:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    if number == 4246:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_small", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    if number == 4247:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    if number == 4248:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    if number == 4249:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    #openvino
    if number == 4290:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    if number == 4291:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)
    if number == 4292:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntu_s", "-num_thr", "4"], shell=False)

    # yolov5l 3 

    # num_threads 3
    if number == 3600:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3601:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3602:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3603:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5s-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 3604:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3605:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3606:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3607:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 3608:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3609:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3610:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3611:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 3612:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3613:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5m", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3614:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3615:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5s", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #yolov8 3 threads
    if number == 3616:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8l_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3617:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8m_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3618:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8n_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3619:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8s_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 3620:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3621:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3622:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3623:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 3624:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3625:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3626:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3627:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 3628:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8l.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3629:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8l.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3630:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8l.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3631:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8l.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #yolov7 3 threads    
    if number == 3632:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov7-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3633:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov7-tiny-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    if number == 3634:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov7.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3635:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov7-tiny.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    if number == 3636:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov7.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3637:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov7-tiny.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    if number == 3638:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov7.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3639:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov7-tiny.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #yolov3 3 threads
    if number == 3640:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov3-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3641:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov3-tiny_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 3642:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov3.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3643:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov3-tiny.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    if number == 3644:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov3.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3645:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov3-tiny.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    if number == 3646:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov3", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 3647:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov3-tiny", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    


    

    # num_threads 4
    if number == 4600:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4601:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4602:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4603:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5s-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 4604:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4605:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4606:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4607:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 4608:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4609:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4610:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4611:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 4612:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4613:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5m", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4614:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4615:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5s", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    
    #yolov8 3 threads
    if number == 4616:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8l_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4617:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8m_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4618:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8n_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4619:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8s_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 4620:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4621:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4622:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4623:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 4624:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4625:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4626:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4627:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 4628:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8l.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4629:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8l.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4630:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8l.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4631:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8l.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    #yolov7 3 threads    
    if number == 4632:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov7-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4633:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov7-tiny-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    
    if number == 4634:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov7.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4635:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov7-tiny.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    
    if number == 4636:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov7.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4637:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov7-tiny.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    
    if number == 4638:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov7.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4639:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov7-tiny.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    #yolov3 3 threads
    if number == 4640:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov3-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4641:
        print("thread 3 testing tflite")
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov3-tiny_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 4642:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov3.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4643:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov3-tiny.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    
    if number == 4644:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov3.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4645:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov3-tiny.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    
    if number == 4646:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov3", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 4647:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov3-tiny", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)




if __name__ == "__main__":
    main()
