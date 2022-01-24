import PIL
from PIL import Image
import pyarmnn as ann
import numpy as np
import argparse
import imageio
import time
import os 
import cv2
print(f"Working with ARMNN {ann.ARMNN_VERSION}")

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

def write_txt_file(model_dict):
  file = open("pyarmnn_avg_time.txt", "w")

  for model, avg_time in model_dict.items():
    file.write(f"{model}, {avg_time}\n")

  file.close()

def inf_pyarmnn(model_path, length=50):
    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(model_path)

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
    opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    graph_id = 0
    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
    input_tensor_id = input_binding_info[0]
    input_tensor_info = input_binding_info[1]
    width, height = input_tensor_info.GetShape()[1], input_tensor_info.GetShape()[2]
    #print(f"tensor id: {input_tensor_id},tensor info: {input_tensor_info}")

    image = np.random.randint(0,255, size=(height,width,3))

    # Get output binding information for an output layer by using the layer name.
    output_names = parser.GetSubgraphOutputTensorNames(graph_id)
    output_binding_info = parser.GetNetworkOutputBindingInfo(0, output_names[0])
    output_tensors = ann.make_output_tensors([output_binding_info])


    net_id, _ = runtime.LoadNetwork(opt_network)
    tot_time = 0

    for i in range(length):
        if ann.TensorInfo.IsQuantized(input_tensor_info):
            image = np.uint8(image)
        else:
            image = np.float32(image/255)

        #print(f"Loaded network, id={net_id}")
        input_tensors = ann.make_input_tensors([input_binding_info], [image])
        start_int_time = time.time()
        runtime.EnqueueWorkload(0, input_tensors, output_tensors)
        tot_time = tot_time + (time.time() - start_int_time)
        results = ann.workload_tensors_to_ndarray(output_tensors)
        print(f"Time: {time.time() - start_int_time}")

    
    print(f"Tot time: {tot_time}, Avg time: {(tot_time)/length}")
    return {model: tot_time/length} 
    

length = 50

dir_path = os.path.dirname(os.path.abspath(__file__))
files_in_path = sorted(os.listdir(dir_path))
models = [p for p in files_in_path if ".tflite" in p]
print(models)
print(dir_path)

result_dict = {}

for model in models:
    model_path = os.path.join(dir_path, model)
    result_dict.update(inf_pyarmnn(model_path, length=length))
    
print(result_dict)
write_txt_file(result_dict)
