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
    # accuracy classification
    #classification tflite
    if number == 1312500:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1312501:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1312502:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    if number == 1312503:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1312504:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1312505:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #classification armnn delegate
    if number == 1310500:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310501:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310502:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    if number == 1310503:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310504:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310505:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #classification onnx
    if number == 1314550:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1" "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1314551:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_075_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1314552:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_050_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1314553:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1314554:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12-int8.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1",  "-os", "ubuntus", "-num_thr", "3"], shell=False)

    # classification pytorch
    if number == 13185150:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 13185151:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 13185152:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_small", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 13185153:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 13185154:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1","-os", "ubuntus", "-num_thr", "3"], shell=False)

    #classification, openvino 
    if number == 13165100:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 13165101:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 13165102:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #latency runs 
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

    #delegate latency runs
    if number == 13120000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310001:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    if number == 1310003:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310004:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1310005:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #onnx latency runs
    if number == 1314050:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so" "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1314051:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_075_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1314052:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_050_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1314053:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1314054:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12-int8.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so",  "-os", "ubuntus", "-num_thr", "3"], shell=False)

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

    #opnvino latency runs 
    if number == 13160100:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 13160101:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 13160102:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #thread testing tf for 1,2,3,4 threads 
    if number == 1112000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1212000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1312000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1412000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1112001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1212001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1312001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1412001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1112002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1212002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1312002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1412002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    
    if number == 1112003:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1212003:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1312003:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1412003:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1112004:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1212004:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1312004:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1412004:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1112005:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1212005:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1312005:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1412005:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    #thread testing for armnn delegate for 1,2,3,4
    if number == 1110000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1210000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1310000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1410000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1110001:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1210001:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1310001:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1410001:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1110002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1210002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1310002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1410002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    
    if number == 1110003:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1210003:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1310003:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1410003:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1110004:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1210004:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1310004:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1410004:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1110005:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1210005:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1310005:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1410005:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    #thread testing tf for no num_thread
    if number == 1012000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus"], shell=False)
    if number == 1012001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus"], shell=False)
    if number == 1012002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus"], shell=False)
    if number == 1012003:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus"], shell=False)
    if number == 1012004:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus"], shell=False)
    if number == 1012005:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus"], shell=False)


    #thread testing for armnn delegate for no num_thread
    if number == 1010000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus"], shell=False)
    if number == 1010001:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus"], shell=False)
    if number == 1010002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus"], shell=False)
    if number == 1010003:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus"], shell=False)
    if number == 1010004:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus"], shell=False)
    if number == 1010005:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus"], shell=False)


    # onnx thread testing for 3,4
    if number == 1314050:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so" "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1414050:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so" "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1314051:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_075_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1414051:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_075_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1314052:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_050_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1414052:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_050_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1314053:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1414053:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1314054:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12-int8.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so",  "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1414054:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12-int8.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so",  "-os", "ubuntus", "-num_thr", "4"], shell=False)

    # pytorch thread testing 3,4
    if number == 13180150:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 14180150:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 13180151:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 14180151:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 13180152:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_small", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 14180152:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_small", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 13180153:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 14180153:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 13180154:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 14180154:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    #openvino thread testing 3,4
    if number == 13160100:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 14160100:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 13160101:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 14160101:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 13160102:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 14160102:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-so", "-os", "ubuntus", "-num_thr", "3"], shell=False)



    
    # cprofiler test on tf
    if number == 13120001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "10", "-os", "ubuntus", "-num_thr", "3", "-p", "cprofiler"], shell=False)
    
    # onnx profiler test
    if number == 13140501:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "10", "-p", "onnx", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #pytorch profiler test
    if number == 131801501:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1", "-ri", "10", "-os", "ubuntus", "-num_thr", "3", "-p", "pytorch"], shell=False)




    sys.exit()

        

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
