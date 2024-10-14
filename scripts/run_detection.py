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
    if number == 1322502:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322503:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5s-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #delegate
    if number == 1320500:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1320501:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1320502:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1320503:
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
    if number == 1326550:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1326551:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1326552:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1326553:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #pytorch
    if number == 1328575:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1328576:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5m", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1328577:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1328578:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5s", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017",  "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #yolov5 latency runs
    #tf
    if number == 1022000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1122000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1222000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1322000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1422000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)


    if number == 1022001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1122001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1222001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1322001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1422001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1022002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1122002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1222002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1322002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1422002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1322003:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5s-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    #delegate
    if number == 1020000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1120000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1220000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1320000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1420000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)


    if number == 1320001:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1020002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1120002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1220002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1320002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1420002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1320003:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5s-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000",  "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #onnx
    if number == 1024025:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1124025:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1224025:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1324025:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1424025:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1324026:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1024027:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1124027:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1224027:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1324027:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1424027:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1324028:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #openvino
    if number == 10326050:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 11326050:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 12326050:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 13326050:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 14326050:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)


    if number == 1326051:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1026052:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1126052:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1226052:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1326052:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1426052:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1326053:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #pytorch
    if number == 10328075:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 11328075:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 12328075:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 13328075:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 14328075:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)


    if number == 1328076:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5m", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1028077:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1128077:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1228077:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)
    if number == 1328077:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1428077:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    if number == 1328078:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5s", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    #yolov3 latency testing
    if number == 1322004:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov3-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322005:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov3-tiny_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1320004:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov3-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1320005:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov3-tiny_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1324029:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov3.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324030:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov3-tiny.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1326054:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov3.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1326055:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov3-tiny.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1328079:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov3", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1328080:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov3-tiny", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)


    #yolov7 latency testing
    if number == 1322006:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov7-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1322007:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov7-tiny-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1320006:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov7-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1320007:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov7-tiny-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1324031:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov7.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324032:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov7-tiny.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1326056:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov7.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1326057:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov7-tiny.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    if number == 1328081:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov7.pt", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1328082:
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

    if number == 1320008:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov8l_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1320009:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov8m_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1320010:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov8n_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1320011:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov8s_float16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1324033:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324034:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324035:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1324036:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov8s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1326058:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1326059:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1326060:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1326061:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov8s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)

    if number == 1328083:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1328084:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8m", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1328085:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    if number == 1328086:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov8s", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "3"], shell=False)
    
    # 4 thread run for latency yolov5
    #tf
    if number == 1422000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1422001:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1422002:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1422003:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5s-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    #delegate
    if number == 1420000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1420001:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5m-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1420002:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5n-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1420003:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5s-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000",  "-os", "ubuntus", "-num_thr", "4"], shell=False)
    
    #onnx
    if number == 1424025:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1","thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1424026:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5m.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1424027:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5n.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1424028:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5s.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    
    #openvino
    if number == 1426050:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1426051:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5m.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1426052:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5n.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1426053:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5s.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    
    #pytorch
    if number == 1428075:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1428076:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5m", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1428077:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5n", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)
    if number == 1428078:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5s", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "4"], shell=False)

    #load testing
    #tf
    if number == 1022000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1122000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1222000:
        subprocess.call(["python3", "run_inference.py", "-api", "tf", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)

    #delegate
    if number == 1020000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1120000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1220000:
        subprocess.call(["python3", "run_inference.py", "-api", "delegate", "-t", "det", "-m", "yolov5l-fp16.tflite", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25","-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)

    #onnx
    if number == 1024025:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1124025:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1224025:
        subprocess.call(["python3", "run_inference.py", "-api", "onnx", "-t", "det", "-m", "yolov5l.onnx", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)

    #openvino
    if number == 1026050:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1126050:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1226050:
        subprocess.call(["python3", "run_inference.py", "-api", "ov", "-t", "det", "-m", "yolov5l.xml", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)

    #pytorch
    if number == 1028075:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus"], shell=False)
    if number == 1128075:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "1"], shell=False)
    if number == 1228075:
        subprocess.call(["python3", "run_inference.py", "-api", "pytorch", "-t", "det", "-m", "yolov5l", "-imgp", "/home/pi/sambashare/BacArbeit/cocoval2017/", "-l", "yolo_labels.txt", "-opd", "-ni", "1", "-thres", "0.25", "-so", "-ri", "1000", "-os", "ubuntus", "-num_thr", "2"], shell=False)

if __name__ == "__main__":
    main()