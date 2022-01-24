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



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', help='File path of .tflite file.', required=True)
args = parser.parse_args()



parser = ann.ITfLiteParser()
network = parser.CreateNetworkFromBinaryFile(args.model)

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
print(f"tensor id: {input_tensor_id},tensor info: {input_tensor_info}")

random_img_list = []
for i in range(50):
    random_img_list.append(np.random.randint(0,255, size=(width,height,3)))

# Get output binding information for an output layer by using the layer name.
output_names = parser.GetSubgraphOutputTensorNames(graph_id)
output_binding_info = parser.GetNetworkOutputBindingInfo(0, output_names[0])
output_tensors = ann.make_output_tensors([output_binding_info])


net_id, _ = runtime.LoadNetwork(opt_network)

start_tot_time = time.time()

for image in random_img_list:
    start_int_time = time.time()
    if ann.TensorInfo.IsQuantized(input_tensor_info):
        image = np.uint8(image)
    else:
        image = np.float32(image/255)

    #print(f"Loaded network, id={net_id}")
    input_tensors = ann.make_input_tensors([input_binding_info], [image])


    runtime.EnqueueWorkload(0, input_tensors, output_tensors)
    results = ann.workload_tensors_to_ndarray(output_tensors)
    max_index = np.where(results[0][0] == np.amax(results[0][0]))
    #print(f"Index: {max_index[0][0]} Pred: {labels[max_index[0][0]]}")
    end_int_time = time.time()
    print(f"Time: {end_int_time - start_int_time}")
    

end_tot_time = time.time()
print(f"Tot time: {end_tot_time-start_tot_time}, Avg time: {(end_tot_time-start_tot_time)/50}")
