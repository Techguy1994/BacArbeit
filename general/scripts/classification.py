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
            for key,value in results[i].items():
                row.append(label_list[key])
                row.append(key)
                row.append(value)
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

def return_n_biggest_result_pyarmnn(output_data, n_big=3):

    max_positions = np.argpartition(output_data[0], -n_big)[-n_big:]

    if output_data.dtype == "uint8":
        out_normalization_factor = np.iinfo(output_data.dtype).max
    elif output_data.dtype == "float32":
        out_normalization_factor = 1
    
    result = {}

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[0][entry] / out_normalization_factor
        result[entry] = [val*100]
        print("\tpos {} : {:.2f}%".format(entry, val*100))
        

    return result

def return_n_biggest_result_tflite_runtime(output_data, output_details, n_big=10):
    max_positions = np.argpartition(output_data[0], -n_big)[-n_big:]
    out_normalization_factor = 1

    print(output_details[0]["dtype"])

    if output_details[0]['dtype'] == np.uint8:
        print("int")
        out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    elif output_details[0]['dtype'] == np.float32:
        print("float")
        out_normalization_factor = 1

    result = {}

    print(out_normalization_factor)

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[0][entry] / out_normalization_factor
        result[entry] = [val*100]
        print("\tpos {} : {:.2f}%".format(entry, val*100))
        
    return result

def inf_tflite_runtime(model_path, pictures_list, n_iter, n_big):
    #source: 
    #https://www.tensorflow.org/lite/guide/inference
    #https://github.com/NXPmicro/pyarmnn-release/tree/master/python/pyarmnn/examples

    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']

    results = []
    inf_times = []
    
    for i in range(n_iter):
        for pic in pictures_list:
            
            img = load_image(input_shape[1], input_shape[2], pic)

            if input_type == np.uint8:
                print(np.iinfo(input_type).max)
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

            #print(output_details[0])
            output_data = interpreter.get_tensor(output_details[0]['index'])
            result = return_n_biggest_result_tflite_runtime(output_data, output_details, n_big)
            results.append(result)

    return inf_times, results
    

def inf_pyarmnn(model_path, pictures_list, n_iter, n_big, inf_times_path):
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

    print(input_tensor_info)

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
            result = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
            result = return_n_biggest_result_pyarmnn(result[0], n_big)
            results.append(result)

    #Profiler Data 
    inf_times = print_profiling_data_and_return_times(profiler)
    write_profiling_data(profiler, model_path, inf_times_path)

    return inf_times, results




if __name__ == "__main__":

    pictures_list = []
    model_list = []
    label = []

    parser = argparse.ArgumentParser(description="Classification inference")
    parser.add_argument("-m", "--model_path", help="one tflite model", required=False)
    parser.add_argument("-mf", "--model_folder_path", default="/home/ubuntu2104/pyarmnn/general/models/classification_models", help="one tflite model", required=False)
    parser.add_argument("-p", "--picture_path", help="one picture for inference", required=False)
    parser.add_argument("-pf", "--picture_folder_path", default="/home/ubuntu2104/pyarmnn/general/input/classification_input", help="one picture for inference", required=False)
    parser.add_argument("-s", '--sleep', type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-l", "--label_path", default="/home/ubuntu2104/pyarmnn/general/models/classification_models/labels/imagenet_labels.txt", help="label folder of the model")
    #parser.add_argument("-lf", "--label_folder_path", help="label folder of the model")
    parser.add_argument("-n", '--niter', default=1, type=int, help='number of iterations', required=False)
    parser.add_argument("-rdtf", '--report_dir_tflite', default='/home/ubuntu2104/pyarmnn/general/results/classification/tflite', help='Directory to save tflite_runtime reports into', required=False)
    parser.add_argument("-rdpy", '--report_dir_pyarmnn', default='/home/ubuntu2104/pyarmnn/general/results/classification/pyarmnn', help='Directory to save pyarmnn reports into', required=False)
    parser.add_argument("--pyarmnn", dest="pyarmnn", action="store_true")
    parser.add_argument("-tflite","--tflite_runtime", dest="tflite_runtime", action="store_true")
    parser.add_argument("--n_big", default=3)
    parser.add_argument("-inf", '--times_pyarmnn', default="/home/ubuntu2104/pyarmnn/general/inf_times/classification", help='Path where the inference data from the pyarmmn profiler is stored ', required=False)
    args = parser.parse_args()

    n_iter = args.niter
    label = args.label_path
    n_big = args.n_big
    inf_times_path = args.times_pyarmnn
    report_dir_tflite = args.report_dir_tflite
    report_dir_pyarmnn = args.report_dir_pyarmnn
    label_list = load_labels(args.label_path)

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
            inf_times, results = inf_pyarmnn(model, pictures_list, n_iter, n_big, inf_times_path)
            model_name = str(model.split("/")[-1].split(".tflite")[0]) + ".csv"
            write_to_csv_file(inf_times, results, label_list, os.path.join(report_dir_pyarmnn, model_name) )
    else:   
        print("no pyarmnn")

    if args.tflite_runtime:
        print("tflite runtime path")
        for model in model_list:
            inf_times, results = inf_tflite_runtime(model, pictures_list, n_iter, n_big)
            model_name = str(model.split("/")[-1].split(".tflite")[0]) + ".csv"
            write_to_csv_file(inf_times, results, label_list, os.path.join(report_dir_tflite, model_name) )
    else:
        print("no tlfite")

