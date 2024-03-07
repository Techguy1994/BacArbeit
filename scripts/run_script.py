import subprocess
import argparse
import sys

def main():
    #dic = create_dictionary()
    handle_arguments()
    

def handle_arguments():
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module for Classification')

        #basic settings
    parser.add_argument("-n", '--n', help='inference API: pyarmnn, tf, ov, onnx, pytorch', required=True) #which api to use
    args = parser.parse_args()
    number = int(args.n)

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




    

    if number == 170:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/Users/marounel-chayeb/BacArbeit/ILSVRC_val/", "-l", "imagenet_classes.txt", "-opd", "-ni", "1"], shell=False)
    if number == 180:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2.xml", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1"], shell=False)
    if number == 181:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2.xml", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "1", "-so"], shell=False)
    if number == 182:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2.xml", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so","-ri", "50"], shell=False)
    if number == 190:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd", "-l", "imagenet_classes.txt", "-opd", "-ni", "4", "-so", "-ri", "50"], shell=False)

    #tflite and pyarmnn inference classification
    #tflite
    if number == 200:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 201:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 202:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 203:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 204:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 205:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)

    #pyarmnn
    if number == 206:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 207:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_uint8_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 208:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_fp32_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 209:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v3_small_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 210:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 211:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_uint8_1.tflite", "-imgp","/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1"], shell=False)
    if number == 212:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 213:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv2-12-int8.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 214:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_large_100_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 215:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_075_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 216:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "class", "-m", "mobilenetv3_small_050_Opset17.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/",  "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    #pytroch
    if number == 217:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 218:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_small", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 219:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 220:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v3_large_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 221:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "class", "-m", "mobilenet_v2_q", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    #openvino
    if number == 222:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 223:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 224:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000//", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 225:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 226:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 227:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v2-1.4-224_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)

    if number == 250:
        subprocess.call(["python3", "run_inference.py", "-api", "pyarmnn", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000"], shell=False)
    if number == 251:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000"], shell=False)
    #comparision tf and delegate when using different number of threads for mobilenetv3 large
    if number == 252:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "1"], shell=False)
    if number == 253:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "2"], shell=False)
    if number == 254:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "3"], shell=False)
    if number == 255:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "4"], shell=False)
    if number == 256:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "1"], shell=False)
    if number == 257:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "2"], shell=False)
    if number == 258:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "3"], shell=False)
    if number == 259:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "4"], shell=False)
    #comparision tf and delegate when using different number of threads for mobilenetv2 large
    if number == 260:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "1"], shell=False)
    if number == 261:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "2"], shell=False)
    if number == 262:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "3"], shell=False)
    if number == 263:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "class", "-m", "lite-lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "4"], shell=False)
    if number == 264:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "1"], shell=False)
    if number == 265:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "2"], shell=False)
    if number == 266:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "3"], shell=False)
    if number == 267:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-lite-model_mobilenet_v2_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000", "-num_thr", "4"], shell=False)

    #tflite latnecy tests
    if number == 270:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "class", "-m", "lite-model_mobilenet_v3_large_100_224_fp32_1.tflite", "-imgd",  "-l", "mobilenet_tflite.txt", "-opd", "-ni", "1", "-ri", "1000"], shell=False)
    
    if number == 280:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 281:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-large-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 282:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)
    if number == 283:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "class", "-m", "mobilenet-v3-small-1.0-224-tf_FP16.xml", "-imgp", "/home/pi/sambashare/BacArbeit/ILSVRC_val_10000/", "-l", "imagenet_classes_mobilenet.txt", "-opd", "-ni", "1"], shell=False)

    

    

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


if __name__ == "__main__":
    main()
