import subprocess
import argparse
import sys

def main():
    handle_arguments()
    

def handle_arguments():
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module for Segmentation')

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
                lite-model_deeplabv3-mobilenetv2_1_default_1:               00
                lite-model_deeplabv3-mobilenetv2-int8_1_default_1.tflite:   01
            
            onnx: 
                lite-model_deeplabv3-mobilenetv2_1_default_1:               25

            openvino:
                deeplabv3_FP32:                                             50


            pytorch:
                deeplabv3_mobilenet_v3_large:                               75

    """

    #deeplabv3 accuracy runs
    #tf
    if number == 1332500:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1332501:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2-int8_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #delegate
    if number == 1330500:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1330501:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2-int8_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #onnx
    if number == 134525:
         subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1"], shell=False)
    
    #openvino
    if number == 136550:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "seg", "-m", "deeplabv3_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #pytorch
    if number == 1338575:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/",  "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #deeplabv3 latency runs
    #no load
    #tf
    if number == 1032500:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os","-so", "-ri", "1000", "ubuntus","-so", "-ri", "1000"], shell=False)
    if number == 1132500:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os","-so", "-ri", "1000", "ubuntus","-so", "-ri", "1000" "-num_thr", "1"], shell=False)
    if number == 1232500:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os","-so", "-ri", "1000", "ubuntus","-so", "-ri", "1000" "-num_thr", "2"], shell=False)
    if number == 1332500:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os","-so", "-ri", "1000", "ubuntus","-so", "-ri", "1000" "-num_thr", "3"], shell=False)
    if number == 1432500:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os","-so", "-ri", "1000", "ubuntus","-so", "-ri", "1000" "-num_thr", "4"], shell=False)


    if number == 1332501:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2-int8_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os","-so", "-ri", "1000", "ubuntus","-so", "-ri", "1000" "-num_thr", "3"], shell=False)

    #delegate 
    if number == 1030500:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l","pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", ,"-so", "-ri", "1000"], shell=False)
    if number == 1130500:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", ,"-so", "-ri", "1000", "-num_thr", "3"], shell=False)
    if number == 1230500:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", ,"-so", "-ri", "1000", "-num_thr", "3"], shell=False)
    if number == 1330500:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", ,"-so", "-ri", "1000", "-num_thr", "3"], shell=False)
    if number == 1430500:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", ,"-so", "-ri", "1000", "-num_thr", "3"], shell=False)

    if number == 1330501:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2-int8_1_default_1.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", ,"-so", "-ri", "1000", "-num_thr", "3"], shell=False)

    #onnx
    if number == 134525:
         subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus","-so", "-ri", "1000"], shell=False)
    if number == 134525:
         subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus","-so", "-ri", "1000", "num_thr", "1"], shell=False)
    if number == 134525:
         subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus","-so", "-ri", "1000", "num_thr", "2"], shell=False)
    if number == 134525:
         subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus","-so", "-ri", "1000", "num_thr", "3"], shell=False)
    if number == 134525:
         subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "seg", "-m", "lite-model_deeplabv3-mobilenetv2_1_default_1.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus","-so", "-ri", "1000", "num_thr", "4"], shell=False)
    
    #openvino
    if number == 136550:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "seg", "-m", "deeplabv3_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus","-so", "-ri", "1000"], shell=False)
    if number == 136550:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "seg", "-m", "deeplabv3_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus","-so", "-ri", "1000", "-num_thr", "1"], shell=False)
    if number == 136550:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "seg", "-m", "deeplabv3_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus","-so", "-ri", "1000", "-num_thr", "2"], shell=False)
    if number == 136550:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "seg", "-m", "deeplabv3_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus","-so", "-ri", "1000", "-num_thr", "3"], shell=False)
    if number == 136550:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "seg", "-m", "deeplabv3_FP32.xml", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/", "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus","-so", "-ri", "1000", "-num_thr", "4"], shell=False)

    #pytorch
    if number == 1338575:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/",  "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", ,"-so", "-ri", "1000"], shell=False)
    if number == 1338575:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/",  "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", ,"-so", "-ri", "1000", "-num_thr", "1"], shell=False)
    if number == 1338575:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/",  "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", ,"-so", "-ri", "1000", "-num_thr", "2"], shell=False)
    if number == 1338575:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/",  "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", ,"-so", "-ri", "1000", "-num_thr", "3"], shell=False)
    if number == 1338575:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "seg", "-m", "deeplabv3_mobilenet_v3_large", "-imgp", "/home/pi/sambashare/BacArbeit/pascal_voc_2012/",  "-l", "pascal_voc_labels.txt", "-c", "pascal_voc_2012","-opd", "-ni", "1", "-os", "ubuntus", ,"-so", "-ri", "1000", "-num_thr", "4"], shell=False)

if __name__ == "__main__":
    main()