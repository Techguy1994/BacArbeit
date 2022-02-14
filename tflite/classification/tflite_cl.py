import numpy as np
import tflite_runtime.interpreter as tflite
from time import sleep, time  
import cv2 
import csv



def load_image(shape):
    
    img = cv2.imread("test.jpg")
    img = cv2.resize(img, (shape[1], shape[2]))
    #cv2.imwrite("Out.jpg", img)
    #print()
    #print(img)
    #print()
    img = np.expand_dims(img, axis=0)
    return img



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

    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="mobilenet_v1_0.5_224_quant.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_type =np.iinfo(input_details[0]['dtype'])

    results = []
    inf_times = []

    for i in range(iter):

        img = load_image(input_shape)
        #print(img.shape)
        #print(img)
        #print()
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
        output_data = interpreter.get_tensor(output_details[0]['index'])
        result = return_n_biggest_result(output_data,output_details, n_big)
        results.append(result)

    return inf_times, results



if __name__ == "__main__":

    n_big = 3
    iter = 5
    inf_times, results = inf_tflite(iter, n_big)
    write_to_csv_file(inf_times, results)

