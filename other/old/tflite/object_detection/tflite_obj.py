import numpy as np
import tflite_runtime.interpreter as tflite
from time import sleep, time   
import csv
import cv2

def load_image(shape):
    
    img = cv2.imread("pede.jpg")
    print(img)
    img = cv2.resize(img, (shape[1], shape[2]))
    cv2.imwrite("Out.jpg", img)
    #print()
    #print(img)
    #print()
    img = np.expand_dims(img, axis=0)
    return img

def draw_rect(output_data, input_shape, img):

    _, height, width, _ = input_shape
    #print(output_data[0][0][0][0])

    print(img.shape)
    img = img[0,:,:]
    print(img.shape)

    for i in range(int(output_data[3][0])):
        print(f"Result: {i+1}")
        print("Printing the rectangle on the image")
        y_min = int(max(1, (output_data[0][0][i][0] * height)))
        x_min = int(max(1, (output_data[0][0][i][1] * width)))
        y_max = int(min(height, (output_data[0][0][i][2] * height)))
        x_max = int(min(width, (output_data[0][0][i][3] *width)))
        print(f"Rect: y_min: {y_min}, x_min: {x_min}, y_max: y_max: {y_max}, x_max: {x_max}")
        print(f"Class index: {output_data[1][0][i]}")
        print(f"score: {output_data[2][0][i]}")

        if output_data[2][0][i] > 0:
            #img = cv2.imread("pede.jpg")
            #img = cv2.resize(img, (300, 300))
            print("Start")
            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
            #print(img)
            cv2.imwrite("tst.jpg", img)
            print("End")




def write_to_csv_file(inf_times, results):
    with open("tst.csv", 'w', newline='\n') as csvfile:
        infwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in range(len(inf_times)):
            row = [f"Inf number: {i+1}", inf_times[i]]
            for result in results[i]:
                row.append(result)
            infwriter.writerow(row)


def return_n_biggest_result(output_data, output_details, n_big=10):
    max_positions = np.argpartition(output_data[0], -n_big)[-n_big:]
    out_normalization_factor = np.iinfo(output_details[0]['dtype']).max

    result = []

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[0][entry] / out_normalization_factor
        result.append(val*100)
        print("\tpos {} : {:.2f}%".format(entry, val*100))
    return result

def inf_tflite(iter=10, n_big=5):
    #source: https://www.tensorflow.org/lite/guide/inference
    #https://github.com/joonb14/TFLiteDetection
    #https://heartbeat.comet.ml/running-tensorflow-lite-object-detection-models-in-python-8a73b77e13f8
    

    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="ssd_mobilenet_v1_1_default_1.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    #print(input_details[0]['dtype'])
    #print(np.iinfo(input_details[0]['dtype']))
    input_type =np.iinfo(input_details[0]['dtype'])
    #print(input_type)

    #arr = np.zeros(iter,n_big)

    results = []
    inf_times = []

    for i in range(iter):

        #format for output_data: [Locations, classes, scores, number of Detections]
        output_data = []

        # loading image 
        img = load_image(input_shape)

        input_data = np.array(img, dtype=input_type)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        beg = time()
        interpreter.invoke()
        end = time()
        inf_time = end-beg
        inf_times.append(inf_time*1000)
        print(inf_time*1000)



        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        print()
        for det in output_details:
            output_data.append(interpreter.get_tensor(det['index']))
            #print(det)
            #print()
        
        
        for out in output_data:
            print(out)
            print()

        draw_rect(output_data, input_shape, img)
        #output_data = interpreter.get_tensor(output_details[0]['index'])
        #print(output_data)
        #result = return_n_biggest_result(output_data,output_details, n_big)
        #results.append(result)

    return inf_times, results



if __name__ == "__main__":

    n_big = 1
    iter = 1
    inf_times, results = inf_tflite(iter, n_big)
    #write_to_csv_file(inf_times, results)

