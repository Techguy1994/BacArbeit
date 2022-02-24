import argparse, os, sys
from time import sleep, time
import numpy as np
import cv2
import csv

import pyarmnn as ann
import tflite_runtime.interpreter as tflite

def return_model_list(models_path):
    model_list = []
    models = os.listdir(models_path)

    for model in models:
        if ".tflite" in model:
            model_list.append(os.path.join(models_path, model))
    
    return model_list

def return_picture_list(pictures_path):
    pictures_list = []
    pictures = os.listdir(pictures_path)

    for picture in pictures:
        if any(end in picture for end in [".jpg", ".png"]):
            pictures_list.append(os.path.join(pictures_path, picture))
        
    return pictures_list

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

def load_image(height, width, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (height, width))
    cv2.imwrite("resized_input.jpg", img)
    img = np.expand_dims(img, axis=0)
    return img

def write_txt_file(model_dict):
  file = open("pyarmnn_avg_time.txt", "w")

  for model, avg_time in model_dict.items():
    file.write(f"{model}, {avg_time}\n")

  file.close()


def write_to_csv_file(inf_times, results, label_list ,csv_path):

    with open(csv_path, 'w', newline='\n') as csvfile:
        infwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in range(len(inf_times)):
            row = [f"Inf number: {i+1}", inf_times[i]]
            print(results[i])
            #for key,value in results[i].items():
            #    row.append(label_list[key])
            #   row.append(key)
            #    row.append(value)
            for n in range(len(results[i][0])):
                row.append(label_list[results[i][0][n]])
                row.append(results[i][0][n])
                row.append(results[i][1][n])
            infwriter.writerow(row)


def setup_profiling(net_id, runtime):
    profiler = runtime.GetProfiler(net_id)
    profiler.EnableProfiling(True)
    return profiler

def print_profiling_data_and_return_times(profiler):

    profiler_data = ann.get_profiling_data(profiler)
    

    times = profiler_data.inference_data["execution_time"]
    tot_time = 0
    for time in times:
        print(f"inference model time: {round(time/1000, 5)}ms")
        tot_time += time

    avg_time = tot_time / len(times)
 
    print(f"Total_time: {round(tot_time/1000,5)}ms, avg_time: {round(avg_time/1000, 5)}ms")
    return [time/1000 for time in times]

def write_profiling_data(profiler, model_path, csv_path):
    profiler_data = ann.get_profiling_data(profiler)

    if not os.path.isdir(csv_path):
        os.mkdir(csv_path)

    # prepare data to be written in csv
    # inference data
    inference_data = profiler_data.inference_data
    tot_time_unit = inference_data["time_unit"]
    inference_times = inference_data["execution_time"]

    if tot_time_unit == "us":
        inference_times = [round(i /1000, 5) for i in inference_times ]
        tot_time_unit = "ms"
    elif tot_time_unit == "s":
        inference_times = [round(i *1000, 5) for i in inference_times ]
        tot_time_unit = "ms"

    # prepare layer data 
    layer_data = profiler_data.per_workload_execution_data

    model_name = str(model_path.split("/")[-1].split(".tflite")[0])

    with open(os.path.join(csv_path, 'test_inf_times_' + model_name.split(".tflite")[0] + ".csv"), 'w', newline='\n') as csvfile:
        infwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        #write first row and total inference times
        infwriter.writerow([model_name, "inferences: time_in_" + tot_time_unit])
        infwriter.writerow(inference_times)

        # write head of layer inference times
        infwriter.writerow(["Layer", "backend", "time_unit", "layer_time"])

        # write layer inferences
        for key, value in layer_data.items():
            layer = key
            backend = value["backend"]
            time_unit = value["time_unit"]
            execution_time = value["execution_time"]

            if time_unit == "us":
                execution_time = [round(i /1000, 5) for i in execution_time ]
                time_unit = "ms"
            elif time_unit == "s":
                execution_time = [round(i *1000, 5) for i in execution_time ]
                time_unit = "ms"

            csv_array = [layer, backend, time_unit]

            for i in execution_time:
                csv_array.append(i)

            infwriter.writerow(csv_array)

def draw_rect_and_return_result(output_data, height, width, img, score_threshold, labels):

    print(labels)
    #print(output_data[0][0][0][0])

    img = img[0,:,:]

    key = []
    value = []
    result = (key,value)

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

        #threshold als Schwelle
        if output_data[2][0][i] > score_threshold:
            #result[output_data[1][0][i]] = [output_data[2][0][i]]
            key.append(output_data[1][0][i])
            value.append(output_data[2][0][i])
            # cv2 put_TEXT function um den Text in Bild hineinschreiben 
            print("Start")
            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
            img = cv2.putText(img, labels[int(output_data[1][0][i])], (x_min+2, y_max-2),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.imwrite("tst.jpg", img)
            print("End")
    
    return result


def inf_tflite_runtime(model_path, pictures_list, n_iter, score_threshold, labels):
    #source: 
    #https://www.tensorflow.org/lite/guide/inference
    #https://github.com/NXPmicro/pyarmnn-release/tree/master/python/pyarmnn/examples

    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details[0])

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']

    results = []
    inf_times = []
    
    for i in range(n_iter):
        for pic in pictures_list:

            #format for output_data: [Locations, classes, scores, number of Detections]
            output_data = []
            
            img = load_image(input_shape[1], input_shape[2], pic)

            if input_type == np.uint8:
                img = np.uint8(img)
            else:
                img = np.float32(img/np.iinfo("uint8").max)

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
            for det in output_details:
                output_data.append(interpreter.get_tensor(det['index']))

            results.append(draw_rect_and_return_result(output_data, input_shape[1], input_shape[2], img, score_threshold, labels))
            
    return inf_times, results
    

def inf_pyarmnn(model_path, pictures_list, n_iter, score_threshold, inf_times_path, labels):
    # LINK TO CODE: https://www.youtube.com/watch?v=HQYosuy4ABY&t=1867s
    #https://developer.arm.com/documentation/102557/latest
    #file:///C:/Users/Maroun_Desktop_PC/SynologyDrive/Bachelorarbeit/pyarmnn/pyarmnn_doc.html#pyarmnn.IOutputSlot

    print(f"Working with ARMNN {ann.ARMNN_VERSION}")

    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(model_path)

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    
    preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef'), ann.BackendId('GpuAcc')]
    opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    print(f"Preferred Backends: {preferredBackends}\n {runtime.GetDeviceSpec()}\n")
    print(f"Optimizationon warnings: {messages}")

    # get input binding information for the input layer of the model
    graph_id = parser.GetSubgraphCount() - 1
    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
    input_tensor_id = input_binding_info[0]
    input_tensor_info = input_binding_info[1]
    width, height = input_tensor_info.GetShape()[1], input_tensor_info.GetShape()[2]
    print(f"tensor id: {input_tensor_id},tensor info: {input_tensor_info}")

    # Get output binding information for an output layer by using the layer name.
    output_names = parser.GetSubgraphOutputTensorNames(graph_id)

    output_binding_info = []

    for output_name in output_names:
        output_binding_info.append(parser.GetNetworkOutputBindingInfo(graph_id, output_name))
    output_tensors = ann.make_output_tensors(output_binding_info)

    net_id, _ = runtime.LoadNetwork(opt_network)

    # Setup the Profilier for layer and network and inference time 
    profiler = setup_profiling(net_id, runtime)
    
    #inference 
    results = []
    
    for i in range(n_iter):
        for pic in pictures_list:
            image = load_image(width, height, pic)

            if ann.TensorInfo.IsQuantized(input_tensor_info):
                image = np.uint8(image)
            else:
                image = np.float32(image/np.iinfo("uint8").max)

            input_tensors = ann.make_input_tensors([input_binding_info], [image])
            runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
            output_data = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
            results.append(draw_rect_and_return_result(output_data, width, height, image, score_threshold, labels))

    #Profiler Data 
    inf_times = print_profiling_data_and_return_times(profiler)
    write_profiling_data(profiler, model_path, inf_times_path)

    return inf_times, results


if __name__ == "__main__":

    pictures_list = []
    model_list = []
    label = []

    parser = argparse.ArgumentParser(description="object detection inference")
    parser.add_argument("-m", "--model_path", help="one tflite model", required=False)
    parser.add_argument("-mf", "--model_folder_path", default="/home/ubuntu2104/pyarmnn/general/models/detection_models", help="one tflite model", required=False)
    parser.add_argument("-p", "--picture_path", help="one picture for inference", required=False)
    parser.add_argument("-pf", "--picture_folder_path", default="/home/ubuntu2104/pyarmnn/general/input/detection_input", help="one picture for inference", required=False)
    parser.add_argument("-s", '--sleep', type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-l", "--label_path", default="/home/ubuntu2104/pyarmnn/general/models/detection_models/labels/coco_labels.txt", help="label folder of the model")
    #parser.add_argument("-lf", "--label_folder_path", help="label folder of the model")
    parser.add_argument("-n", '--niter', default=1, type=int, help='number of iterations', required=False)
    parser.add_argument("-rdtf", '--report_dir_tflite', default='/home/ubuntu2104/pyarmnn/general/results/object_detection/tflite', help='Directory to save tflite_runtime reports into', required=False)
    parser.add_argument("-rdpy", '--report_dir_pyarmnn', default='/home/ubuntu2104/pyarmnn/general/results/object_detection/pyarmnn', help='Directory to save pyarmnn reports into', required=False)
    parser.add_argument("--pyarmnn", dest="pyarmnn", action="store_true")
    parser.add_argument("-tflite","--tflite_runtime", dest="tflite_runtime", action="store_true")
    parser.add_argument("-thres","--score_threshold", default=0.5, help="specifies the threshold for the score to be drawn and input into the result")
    parser.add_argument("-inf", '--times_pyarmnn', default="/home/ubuntu2104/pyarmnn/general/inf_times/object_detection", help='Path where the inference data from the pyarmmn profiler is stored ', required=False)
    args = parser.parse_args()

    n_iter = args.niter
    label = args.label_path
    thresh = args.score_threshold
    inf_times_path = args.times_pyarmnn
    report_dir_tflite = args.report_dir_tflite
    report_dir_pyarmnn = args.report_dir_pyarmnn
    label_list = load_labels(args.label_path)

    thresh = 0.3

    if(args.model_path):
        model_list.append(args.model_path)
    else:
        model_list = return_model_list(args.model_folder_path)

    if(args.picture_path):
        pictures_list.append(args.picture_path)
    else:
        pictures_list = return_picture_list(args.picture_folder_path)
    
    if args.pyarmnn: 
        print("Pyarmnn path")
        for model in model_list:
            inf_times, results = inf_pyarmnn(model, pictures_list, n_iter, thresh, inf_times_path, label_list)
            model_name = str(model.split("/")[-1].split(".tflite")[0]) + ".csv"
            write_to_csv_file(inf_times, results, label_list, os.path.join(report_dir_pyarmnn, model_name) )
    else:   
        print("no pyarmnn inference")

    if args.tflite_runtime:
        print("tflite runtime path")
        for model in model_list:
            inf_times, results = inf_tflite_runtime(model, pictures_list, n_iter, thresh, label_list)
            model_name = str(model.split("/")[-1].split(".tflite")[0]) + ".csv"
            write_to_csv_file(inf_times, results, label_list, os.path.join(report_dir_tflite, model_name))
    else:
        print("no tlfite inference")

