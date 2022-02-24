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

def load_image(width, height, image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (height, width))
    cv2.imwrite("resized_input.jpg", resized_img)
    resized_img = np.expand_dims(resized_img, axis=0)
    return img, resized_img

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def vis_segmentation_cv2(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  print("Start segm")
  #plt.figure(figsize=(15, 5))
  #grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  cv2.imwrite("result1.jpg", image)
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  print(image.shape)
  print(seg_image.shape)
  #seg_image = cv2.cvtColor(seg_image, cv2.COLOR_GRAY2RGB)
  cv2.imwrite("result2.jpg", seg_image)

  print("shapes")
  print(image.shape)
  print(seg_image.shape)
  print("end")

  overlay_picture = cv2.addWeighted(image, 0.7, seg_image, 0.5, 0)
  cv2.imwrite("result3.png", overlay_picture)

  #image = image.convert("RGBA")
  #seg_image = seg_image.convert("RGBA")

  #overlay = Image.blend(image, seg_image, 0.5)
  #overlay.save("result3.png")

  LABEL_NAMES = np.asarray([
      'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
  ])

  FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
  FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


  unique_labels = np.unique(seg_map)
  indeces = FULL_COLOR_MAP[unique_labels].astype(np.uint8)
  res = LABEL_NAMES[unique_labels]
  print(unique_labels)
  print(indeces)
  print(res)

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
            
            img, resized_img = load_image(input_shape[1], input_shape[2], pic)
            print(f"Pic:{img.shape}")
            
            if input_type == np.uint8:
                resized_img = np.uint8(resized_img)
            else:
                #resized_img = np.float32(resized_img/np.iinfo("uint8").max)
                resized_img = np.float32(resized_img / 127.5 - 1)

            input_data = np.array(resized_img, dtype=input_type)
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

            # my method, first resize, then argmax 
            output_data = output_data[0]

            seg_map = np.ndarray((img.shape[0],img.shape[1],len(output_data[0,0,:])))
            print(seg_map.shape)

            for i in range(len(output_data[0,0,:])):
                seg_map[:,:,i] = cv2.resize(output_data[:,:,i], (img.shape[1],img.shape[0]))

            print(seg_map.shape)
            print(seg_map)
            
            seg_map = np.argmax(seg_map, axis=2)
            print(seg_map.shape)

            vis_segmentation_cv2(img, seg_map)

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
            image, resized_image = load_image(width, height, pic)

            if ann.TensorInfo.IsQuantized(input_tensor_info):
                resized_image = np.uint8(resized_image)
            else:
                resized_image = np.float32(resized_image/np.iinfo("uint8").max)

            input_tensors = ann.make_input_tensors([input_binding_info], [resized_image])
            runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
            result = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict

            seg_map = np.uint8(np.argmax(result, axis=3)[0])
            seg_map = np.squeeze(np.argmax(result, axis=3)).astype(np.int8)
            #seg_map = Image.fromarray(seg_map)
            seg_map = cv2.resize(seg_map, (500,475), interpolation=cv2.INTER_NEAREST)

            vis_segmentation_cv2(image, seg_map)
            #result = return_n_biggest_result_pyarmnn(result[0], n_big)
            #results.append(result)

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
    parser.add_argument("-mf", "--model_folder_path", default="/home/ubuntu2104/pyarmnn/general/models/segmentation_models", help="one tflite model", required=False)
    parser.add_argument("-p", "--picture_path", help="one picture for inference", required=False)
    parser.add_argument("-pf", "--picture_folder_path", default="/home/ubuntu2104/pyarmnn/general/input/segmentation_input", help="one picture for inference", required=False)
    parser.add_argument("-s", '--sleep', type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-l", "--label_path", default="/home/ubuntu2104/pyarmnn/general/models/segmentation_models/labels/labels_pascal_voc_2012.txt", help="label folder of the model")
    #parser.add_argument("-lf", "--label_folder_path", help="label folder of the model")
    parser.add_argument("-n", '--niter', default=1, type=int, help='number of iterations', required=False)
    parser.add_argument("-rdtf", '--report_dir_tflite', default='/home/ubuntu2104/pyarmnn/general/results/segmentation/tflite', help='Directory to save tflite_runtime reports into', required=False)
    parser.add_argument("-rdpy", '--report_dir_pyarmnn', default='/home/ubuntu2104/pyarmnn/general/results/segmentation/pyarmnn', help='Directory to save pyarmnn reports into', required=False)
    parser.add_argument("--pyarmnn", dest="pyarmnn", action="store_true")
    parser.add_argument("-tflite","--tflite_runtime", dest="tflite_runtime", action="store_true")
    parser.add_argument("--n_big", default=3)
    parser.add_argument("-inf", '--times_pyarmnn', default="/home/ubuntu2104/pyarmnn/general/inf_times/segmentation", help='Path where the inference data from the pyarmmn profiler is stored ', required=False)
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
            #model_name = str(model.split("/")[-1].split(".tflite")[0]) + ".csv"
            #write_to_csv_file(inf_times, results, label_list, os.path.join(report_dir_pyarmnn, model_name) )
    else:   
        print("no pyarmnn")

    if args.tflite_runtime:
        print("tflite runtime path")
        for model in model_list:
            inf_times, results = inf_tflite_runtime(model, pictures_list, n_iter, n_big)
            #model_name = str(model.split("/")[-1].split(".tflite")[0]) + ".csv"
            #write_to_csv_file(inf_times, results, label_list, os.path.join(report_dir_tflite, model_name) )
    else:
        print("no tlfite")

