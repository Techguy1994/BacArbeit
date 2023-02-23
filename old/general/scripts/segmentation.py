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
    return np.asarray([line.strip() for i, line in enumerate(f.readlines())])

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

def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  return colormap

def create_ade20k_label_colormap():
  """Creates a label colormap used in ADE20K segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])

def label_to_color_image(label, colormap):
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

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def vis_segmentation_cv2(image, seg_map, LABEL_NAMES, colormap):
  """Visualizes input image, segmentation map and overlay view."""

  cv2.imwrite("result1.jpg", image)
  seg_image = label_to_color_image(seg_map, colormap).astype(np.uint8)

  overlay_picture = cv2.addWeighted(image, 0.7, seg_image, 0.5, 0)

  #LABEL_NAMES = np.asarray([
  #    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
  #    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
  #    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
  #])

  #print(LABEL_NAMES)
  #FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
  #FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP, colormap)


  unique_labels = np.unique(seg_map)
  #indeces = FULL_COLOR_MAP[unique_labels].astype(np.uint8)
  res = LABEL_NAMES[unique_labels]
  #print(unique_labels)
  #print(indeces)
  #print(res)

  return res, overlay_picture, seg_image

def write_to_csv_file(inf_times, results, label_list ,csv_path, picture_list):

    with open(csv_path, 'w', newline='\n') as csvfile:
        infwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in range(len(inf_times)):
            row = [f"Inf number: {i+1}", inf_times[i]]
            for value in results[i]:
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


def inf_tflite_runtime(model_path, pictures_list, n_iter, label_list, pic_out_path, colormap):
    #source: 
    #https://www.tensorflow.org/lite/guide/inference
    #https://github.com/NXPmicro/pyarmnn-release/tree/master/python/pyarmnn/examples

    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(label_list) != output_details[0]["shape"][3]:
        sys.exit("label and colormap have not the same length")

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

            for i in range(len(output_data[0,0,:])):
                seg_map[:,:,i] = cv2.resize(output_data[:,:,i], (img.shape[1],img.shape[0]))

    
            seg_map = np.argmax(seg_map, axis=2)

            result, out_image, out_mask = vis_segmentation_cv2(img, seg_map, label_list, colormap)
            results.append(result)
            gen_out_path = os.path.join(pic_out_path, pic.split("/")[-1].split(".")[0])
            mask_out_path = gen_out_path + "_mask.jpg"
            result_pic_out_path = gen_out_path + ".jpg"
            cv2.imwrite(result_pic_out_path, out_image)
            cv2.imwrite(mask_out_path, out_mask)
            
    return inf_times, results, out_mask
    

def inf_pyarmnn(model_path, pictures_list, n_iter, inf_times_path, label_list, pic_out_path, colormap):
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

    if len(label_list) != output_tensors[0][1].GetShape()[3]:
        sys.exit("label and colormap have not the same length")


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
                #resized_image = np.float32(resized_image/np.iinfo("uint8").max)
                resized_image = np.float32(resized_image / 127.5 - 1)

            input_tensors = ann.make_input_tensors([input_binding_info], [resized_image])
            runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
            output_data = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict

            output_data = output_data[0][0]
    
            seg_map = np.ndarray((image.shape[0],image.shape[1],len(output_data[0,0,:])))

            for i in range(len(output_data[0,0,:])):
                seg_map[:,:,i] = cv2.resize(output_data[:,:,i], (image.shape[1],image.shape[0]))


            
            seg_map = np.argmax(seg_map, axis=2)

            result, out_image, out_mask = vis_segmentation_cv2(image, seg_map,label_list, colormap)
            results.append(result)
            gen_out_path = os.path.join(pic_out_path, pic.split("/")[-1].split(".")[0])
            mask_out_path = gen_out_path + "_mask.jpg"
            result_pic_out_path = gen_out_path + ".jpg"
            cv2.imwrite(result_pic_out_path, out_image)
            cv2.imwrite(mask_out_path, out_mask)


    #Profiler Data 
    inf_times = print_profiling_data_and_return_times(profiler)
    write_profiling_data(profiler, model_path, inf_times_path)

    return inf_times, results, out_mask


if __name__ == "__main__":
    #source: https://github.com/tensorflow/models/blob/master/research/deeplab/utils/get_dataset_colormap.py

    general_dir = os.path.abspath(os.path.dirname(__file__)).split("scripts")[0]

    pictures_list = []
    model_list = []
    label = []

    parser = argparse.ArgumentParser(description="Segmentation inference")
    parser.add_argument("-m", "--model_path", help="one tflite model", required=False)
    parser.add_argument("-mf", "--model_folder_path", default=os.path.join(general_dir, "models/segmentation_models"), help="one tflite model", required=False)
    parser.add_argument("-p", "--picture_path", help="one picture for inference", required=False)
    parser.add_argument("-pf", "--picture_folder_path", default=os.path.join(general_dir, "input/segmentation_input"), help="one picture for inference", required=False)
    parser.add_argument("-s", '--sleep', type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-l", "--label_path", default=os.path.join(general_dir, "models/segmentation_models/labels/labelmap.txt"), help="label folder of the model")
    parser.add_argument("-n", '--niter', default=1, type=int, help='number of iterations', required=False)
    parser.add_argument("-rdtf", '--report_dir_tflite', default=os.path.join(general_dir, "results/segmentation/tflite"), help='Directory to save tflite_runtime reports into', required=False)
    parser.add_argument("-rdpy", '--report_dir_pyarmnn', default=os.path.join(general_dir, "results/segmentation/pyarmnn"), help='Directory to save pyarmnn reports into', required=False)
    parser.add_argument("--pyarmnn", dest="pyarmnn", action="store_true")
    parser.add_argument("-tflite","--tflite_runtime", dest="tflite_runtime", action="store_true")
    parser.add_argument("--n_big", default=3)
    parser.add_argument("-inf", '--times_pyarmnn', default=os.path.join(general_dir, "inf_times/segmentation"), help='Path where the inference data from the pyarmmn profiler is stored ', required=False)
    parser.add_argument("-dat", "--dataset", help="which dataset the model/s is/are based on. The ones available are ade20k, pascal_voc_2012, cityscapes", required=True)
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

    if args.dataset == "ade20k":
        print(args.dataset)
        colormap = create_ade20k_label_colormap()
    elif args.dataset == "pascal_voc_2012":
        colormap = create_pascal_label_colormap()
    elif args.dataset == "cityscapes":
        colormap = create_cityscapes_label_colormap()
    else:
        sys.exit("No vaild name for dataset given")

    if args.pyarmnn: 
        print("Pyarmnn path")
        for model in model_list:
            model_name = str(model.split("/")[-1].split(".tflite")[0])
            pic_out_path = os.path.join(report_dir_pyarmnn, model_name)
            if not os.path.isdir(pic_out_path):
                os.mkdir(pic_out_path)  
            inf_times, results, pyarmnn_mask = inf_pyarmnn(model, pictures_list, n_iter, inf_times_path, label_list, pic_out_path, colormap)
            model_name_csv = model_name + ".csv"
            write_to_csv_file(inf_times, results, label_list, os.path.join(report_dir_pyarmnn, model_name_csv), pictures_list)
    else:   
        print("no pyarmnn")

    if args.tflite_runtime:
        print("tflite runtime path")
        for model in model_list:
            model_name = str(model.split("/")[-1].split(".tflite")[0])
            pic_out_path = os.path.join(report_dir_tflite, model_name)
            if not os.path.isdir(pic_out_path):
                os.mkdir(pic_out_path) 
            inf_times, results, tflite_mask = inf_tflite_runtime(model, pictures_list, n_iter, label_list, pic_out_path, colormap)
            model_name_csv = model_name + ".csv"
            write_to_csv_file(inf_times, results, label_list, os.path.join(report_dir_tflite, model_name_csv), pictures_list)
    else:
        print("no tlfite")


