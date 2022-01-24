import PIL
from PIL import Image
import pyarmnn as ann
import numpy as np
import argparse
import imageio
import time
import os
import csv

#import cv2
print(f"Working with ARMNN {ann.ARMNN_VERSION}")

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

def write_txt_file(model_dict):
  file = open("pyarmnn_avg_time.txt", "w")

  for model, avg_time in model_dict.items():
    file.write(f"{model}, {avg_time}\n")

  file.close()

def inf_pyarmnn(model_path, iterations=50):
    # LINK TO CODE: https://www.youtube.com/watch?v=HQYosuy4ABY&t=1867s
    #file:///C:/Users/Maroun_Desktop_PC/SynologyDrive/Bachelorarbeit/pyarmnn/pyarmnn_doc.html#pyarmnn.IOutputSlot
    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(model_path)

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
    opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    print(f"Preferred Backends: {preferredBackends}\n {runtime.GetDeviceSpec()}\n")
    print(f"Optimizationon warnings: {messages}")

    graph_id = parser.GetSubgraphCount() - 1
    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
    input_tensor_id = input_binding_info[0]
    input_tensor_info = input_binding_info[1]
    width, height = input_tensor_info.GetShape()[1], input_tensor_info.GetShape()[2]
    #print(f"tensor id: {input_tensor_id},tensor info: {input_tensor_info}")

    image = np.random.randint(0,255, size=(height,width,3))

    # Get output binding information for an output layer by using the layer name.
    output_names = parser.GetSubgraphOutputTensorNames(graph_id)

    output_binding_info = []

    for output_name in output_names:
        output_binding_info.append(parser.GetNetworkOutputBindingInfo(graph_id, output_name))

    #output_binding_info = parser.GetNetworkOutputBindingInfo(0, output_names[0])
    output_tensors = ann.make_output_tensors(output_binding_info)
    print("\n")
    print(opt_network)
    print("\n")
    net_id, _ = runtime.LoadNetwork(opt_network)
    tot_time = 0
    time_list = []
    time_list.append(str(model_path.split("/")[-1].split(".tflite")[0]))

    time.sleep(0.5)
    for i in range(iterations):
        if ann.TensorInfo.IsQuantized(input_tensor_info):
            image = np.uint8(image)
        else:
            image = np.float32(image/255)

        input_tensors = ann.make_input_tensors([input_binding_info], [image])

        start_int_time = time.time()

        runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
        inf_time = time.time() - start_int_time
        tot_time = tot_time + inf_time

        #results = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict

        print("Inference Time: {0:.3f} ms".format(inf_time*1000))

        #time_list.append("{0:.6f}".format(inf_time))
        time.sleep(0.1)

    time.sleep(0.5)

    print("Tot time: {0:.3f} ms, Avg time: {0:.3f} ms".format(tot_time*1000, 1000*tot_time/iterations))
    return {model: tot_time/iterations}, time_list


if __name__ == "__main__":
    iterations = 10

    dir_path = os.path.dirname(os.path.abspath(__file__))
    files_in_path = sorted(os.listdir(os.path.join(dir_path, "models")))
    models = [p for p in files_in_path if ".tflite" in p]
    print(models)
    #print(dir_path)

    result_dict = {}
    tot_inf_times = []
    model = ""

    """
    model = models[12]
    model_path = os.path.join(dir_path, "models", model)
    avg_dict, inf_times = inf_pyarmnn(model_path, iterations=iterations)
    result_dict.update(avg_dict)
    tot_inf_times.append(inf_times)"""

    for i, model in enumerate(models):
        model_path = os.path.join(dir_path, "models", model)
        avg_dict, inf_times = inf_pyarmnn(model_path, iterations=iterations)
        result_dict.update(avg_dict)
        tot_inf_times.append(inf_times)


    print(result_dict)
    #write_txt_file(result_dict)
    csv_path = "/home/ubuntu2104/pyarmnn/inf_times"
    if not os.path.isdir(csv_path):
        os.mkdir(csv_path)

    with open(os.path.join(csv_path, 'test_inf_times_' + model.split(".tflite")[0] + ".csv"), 'w', newline='\n') as csvfile:
        infwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for row in tot_inf_times:
            infwriter.writerow(row)
