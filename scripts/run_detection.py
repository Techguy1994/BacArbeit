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
                yolov5l:        00
                yolov5m:        01
                yolov5n:        02
                yolov5s:        03
                yolov3:         04
                yolov3-tiny:    05
                yolov7:         06
                yolov7-tiny:    07
                yolov8l:        08
                yolov8m:        09
                yolov8n:        10
                yolov8s:        11
            
            onnx: 
                yolov5l:        25
                yolov5m:        26
                yolov5n:        27
                yolov5s:        28
                yolov3:         29
                yolov3-tiny:    30
                yolov7:         31
                yolov7-tiny:    32
                yolov8l:        33
                yolov8m:        34
                yolov8n:        35
                yolov8s:        36

            openvino:
                yolov5l:        50
                yolov5m:        51
                yolov5n:        52
                yolov5s:        53
                yolov3:         54
                yolov3-tiny:    55
                yolov7:         56
                yolov7-tiny:    57
                yolov8l:        58
                yolov8m:        59
                yolov8n:        60
                yolov8s:        61

            pytorch:
                yolov5l:        75
                yolov5m:        76
                yolov5n:        77
                yolov5s:        78
                yolov3:         79
                yolov3-tiny:    80
                yolov7:         81
                yolov7-tiny:    82
                yolov8l:        83
                yolov8m:        84
                yolov8n:        85
                yolov8s:        86
    """

    #yolov5 
    #yolov5 accuracy
    #tf
    if number == 1322500:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322501:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322503:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322504:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5s-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #delegate
    if number == 1320500:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1320501:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1320503:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1320504:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5s-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #onnx
    if number == 1324525:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324526:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324527:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324528:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #openvino 
    if number == 1236550:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1236551:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1236552:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1236553:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #pytorch
    if number == 1238575:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1238576:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5m", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1238577:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1238578:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5s", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #yolov5 latency runs
    #tf
    if number == 1322000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322003:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5s-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #onnx
    if number == 1324025:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324026:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324027:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324028:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #openvino
    if number == 1236050:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1236051:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1236052:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1236053:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #pytorch
    if number == 1238075:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1238076:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5m", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1238077:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1238078:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5s", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #yolov3 latency testing
    if number == 1322004:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov3-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322005:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov3-tiny_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1324029:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov3.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324030:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov3-tiny.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1236054:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov3.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1236055:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov3-tiny.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1238079:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov3", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1238080:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov3-tiny", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)


    #yolov7 latency testing
    if number == 1322006:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov7-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322007:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov7-tiny-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1324031:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov7.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324032:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov7-tiny.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1236056:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov7.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1236057:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov7-tiny.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    if number == 1238081:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov7.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1238082:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov7-tiny.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #yolov8 latency testing
    if number == 1322008:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8l_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322009:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8m_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322010:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8n_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322011:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov8s_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1324033:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324034:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324035:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324036:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1236058:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1236059:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1236060:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1236061:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1238083:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1238084:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8m", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1238085:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1238086:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8s", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #thread testing



if __name__ == "__main__":
    main()