import torch
from torchvision import models, transforms
import argparse
import logging as log
import os
from time import sleep, time
import numpy as np
import pyarmnn as ann
import tflite_runtime.interpreter as tflite
import cv2
import sys 
from PIL import Image
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type
import onnxruntime
from PIL import Image
import cProfile, pstats, timeit
import csv
import json
from torch.profiler import profile, record_function, ProfilerActivity

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

  print("colormap: --- >", colormap.shape)
  print(colormap)
  print("finto")
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

  print("Label shape: ", label.shape)  

  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')
  print("label to color")
  #print(label)
  #print(colormap)
  print(label.shape, colormap.shape)
  #print(colormap[label])

  return colormap[label]

def vis_segmentation_cv2(image, seg_map, LABEL_NAMES, colormap):
 """Visualizes input image, segmentation map and overlay view."""

#print(image.shape, )
 cv2.imwrite("result1.jpg", image)
 seg_image = label_to_color_image(seg_map, colormap).astype(np.uint8)

 print(image.shape, seg_image.shape)
 print(LABEL_NAMES)

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

def output_data_tflite_runtime(output_details, interpreter, img_res, img_org, img_result_file, img_result_mask_file, class_dir, colormap):
        
    with open(class_dir, 'r') as f:
        labels =  np.asarray([line.strip() for i, line in enumerate(f.readlines())])
    
    #print(output_details[0])
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # my method, first resize, then argmax 
    output_data = output_data[0]

    print(output_data.shape)

    seg_map = np.ndarray((img_res.shape[0],img_res.shape[1],len(output_data[0,0,:])))

    for i in range(len(output_data[0,0,:])):
        seg_map[:,:,i] = cv2.resize(output_data[:,:,i], (img_res.shape[1],img_res.shape[0]))


    seg_map = np.argmax(seg_map, axis=2)

    result, out_image, out_mask = vis_segmentation_cv2(img_res, seg_map, labels, colormap)
    print(img_result_file)
    #results.append(result)
    #gen_out_path = os.path.join(img_result_file, img_res.split("/")[-1].split(".")[0])
    #mask_out_path = gen_out_path + "_mask.jpg"
    #result_pic_out_path = gen_out_path + ".jpg"
    cv2.imwrite(img_result_file, out_image)
    cv2.imwrite(img_result_mask_file, out_mask)

    return result

def output_data_pyarmnn(output_data, img_res, img_org, img_result_file, img_result_mask_file, class_dir, colormap):

    with open(class_dir, 'r') as f:
        labels =  np.asarray([line.strip() for i, line in enumerate(f.readlines())])

    print("label shape", labels.shape)

    #print(output_data[0].shape)
    #print(output_data[0][0].shape)
    #print(output_data[0][0])
    
    output_data = output_data[0][0]

    #print(output_data[0,0,:])
    #print(len(output_data[0,0,:]))
    #print("output_data", output_data.shape)


    #seg_map = np.ndarray((img_res.shape[0],img_res.shape[1],len(output_data[0,0,:])))
    seg_map = np.ndarray((img_res.shape[0],img_res.shape[1],len(output_data[0,0,:])))

    print("segmap", seg_map.shape)

    for i in range(len(output_data[0,0,:])):
        seg_map[:,:,i] = cv2.resize(output_data[:,:,i], (img_res.shape[1],img_res.shape[0]))


    seg_map = np.argmax(seg_map, axis=2)

    print("segmap after argmax (labels)", seg_map.shape)

    result, out_image, out_mask = vis_segmentation_cv2(img_res, seg_map, labels, colormap)
    print(img_result_file)
    #results.append(result)
    #gen_out_path = os.path.join(img_result_file, img_res.split("/")[-1].split(".")[0])
    #mask_out_path = gen_out_path + "_mask.jpg"
    #result_pic_out_path = gen_out_path + ".jpg"
    cv2.imwrite(img_result_file, out_image)
    cv2.imwrite(img_result_mask_file, out_mask)

    return result

def output_data_pytorch_deeplabv3(output_data, img_res, img_org, img_result_file, img_result_mask_file, class_dir, colormap):
    #https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd-mobilenetv1
    results = []

    with open(class_dir, 'r') as f:
        labels =  np.asarray([line.strip() for i, line in enumerate(f.readlines())])

    output = output_data.numpy()
    
    
    #print(output_details[0])
   # output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # my method, first resize, then argmax 
    #output_data = output_data[0]

    print(output_data.shape)
    print(output.shape)
    print(img_res.shape)
    print(len(output_data[:,0,0]))

    seg_map = np.ndarray((img_res.shape[0],img_res.shape[1],len(output[:,0,0])))

    print(seg_map.shape)

    for i in range(len(output[:,0,0])):
        seg_map[:,:,i] = cv2.resize(output[i,:,:], (img_res.shape[1],img_res.shape[0]))


    seg_map = np.argmax(seg_map, axis=2)

    result, out_image, out_mask = vis_segmentation_cv2(img_res, seg_map, labels, colormap)
    print(img_result_file)
    #results.append(result)
    #gen_out_path = os.path.join(img_result_file, img_res.split("/")[-1].split(".")[0])
    #mask_out_path = gen_out_path + "_mask.jpg"
    #result_pic_out_path = gen_out_path + ".jpg"
    cv2.imwrite(img_result_file, out_image)
    cv2.imwrite(img_result_mask_file, out_mask)

    return result

def output_data_onnx(output_data, img_res, img_org, img_result_file, img_result_mask_file, class_dir, colormap):
    #https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd-mobilenetv1
    results = []

    print("label dir", class_dir)

    with open(class_dir, 'r') as f:
        labels =  np.asarray([line.strip() for i, line in enumerate(f.readlines())])

    print(labels.shape)

    
    # my method, first resize, then argmax 
    #output_data = output_data[0]
    print("---------------Output-------------------------------")
    #print(output_data[0][0])
    #print(output_data[0][0].shape)

    output_data = output_data[0][0]


    seg_map = np.ndarray((img_res.shape[0],img_res.shape[1],len(output_data[0,0,:])))

    for i in range(len(output_data[0,0,:])):
        seg_map[:,:,i] = cv2.resize(output_data[:,:,i], (img_res.shape[1],img_res.shape[0]))


    seg_map = np.argmax(seg_map, axis=2)

    result, out_image, out_mask = vis_segmentation_cv2(img_res, seg_map, labels, colormap)
    print(img_result_file)
    #results.append(result)
    #gen_out_path = os.path.join(img_result_file, img_res.split("/")[-1].split(".")[0])
    #mask_out_path = gen_out_path + "_mask.jpg"
    #result_pic_out_path = gen_out_path + ".jpg"
    cv2.imwrite(img_result_file, out_image)
    cv2.imwrite(img_result_mask_file, out_mask)

    return result

def output_data_ov(output_data, img_res, img_org, img_result_file, img_result_mask_file, class_dir, colormap):
    #https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd-mobilenetv1
    results = []

    with open(class_dir, 'r') as f:
        labels =  np.asarray([line.strip() for i, line in enumerate(f.readlines())])

    print("-----------------Output----------------")
    print("colormap: ", colormap.shape)
    print(colormap)
    for element in output_data:
        output = output_data[element][0]
        

    output = np.array(output, dtype="int")


    print(img_res.shape)
    print(img_org.shape, output.shape)

    img_res = cv2.resize(img_res, output.shape)
    print(img_res.shape)
    
    result, out_image, out_mask = vis_segmentation_cv2(img_res, output, labels, colormap)

    out_image = cv2.resize(out_image, (img_org.shape[0], img_org.shape[1]), interpolation=cv2.INTER_AREA)

    print(img_result_file)
    results.append(result)
    cv2.imwrite(img_result_file, out_image)
    cv2.imwrite(img_result_mask_file, out_mask)
    

    return results

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess_image_pytorch_deeplabv3():

    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    return preprocess

def preprocess_image_ov_deeplabv3(input_tensor, model):
    # --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
        print("Preprocess")
        ppp = PrePostProcessor(model)

        a, h, w, z = input_tensor.shape
        print(a, h, w, z)
        print(input_tensor.shape)

        print("Shape out finished")

        # 1) Set input tensor information:
        # - input() provides information about a single model input
        # - reuse precision and shape from already available `input_tensor`
        # - layout of data is 'NHWC'
        ppp.input().tensor() \
            .set_shape(input_tensor.shape) \
            .set_element_type(Type.u8) \
            .set_layout(Layout('NHWC'))  # noqa: ECE001, N400

        # 2) Adding explicit preprocessing steps:
        # - apply linear resize from tensor spatial dims to model spatial dims
        ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)

        # 3) Here we suppose model has 'NCHW' layout for input
        #ppp.input().model().set_layout(Layout('NCHW'))
        ppp.input().model().set_layout(Layout('NHWC'))

        # 4) Set output tensor information:
        # - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(Type.f32)

        # 5) Apply preprocessing modifying the original 'model'
        model = ppp.build()

        return input_tensor, model

def preprocess_image_onnx_deeplabv3(image_path, input_type, image_height, image_width):
    #img = Image.open(image_path)

    #img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], channels)
    #img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
    #return img_data
    #print(input_type.numpy())

    image = cv2.imread(image_path)
    print("---------------------------------------------------------------------------------")
    print(image_height, image_width)
    print(image.shape)
    image = cv2.resize(image, (image_height, image_width))
    image_data = cv2.normalize(image.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    print(image.shape)
    image_data = np.expand_dims(image_data, 0)
    print(image.shape)

    return image_data


def preprocess_image_tflite_deeplabv3(image_path, height, width, input_type, channels=3):
    """
    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS)
    image_data = np.asarray(image).astype(input_type)
    #for channel in range(image_data.shape[0]):
    #    image_data[channel, :, :] = image_data[channel, :, :]*2 / 255 - 1
    image_data = np.expand_dims(image_data, 0)
    #print(image_data.shape)
    #quit()
    return image_data
    """

    image = cv2.imread(image_path)
    image = cv2.resize(image, (width, height))
    image_data = cv2.normalize(image.astype(input_type), None, 0.0, 1.0, cv2.NORM_MINMAX)
    print(image.shape)
    image_data = np.expand_dims(image_data, 0)
    print(image.shape)

    return image, image_data



def setup_profiling(net_id, runtime):
    profiler = runtime.GetProfiler(net_id)
    profiler.EnableProfiling(True)
    return profiler

def check_directories(model_dir, img_dir, model_type):

    if not model_dir:
        quit("Empty model directory")
    if not img_dir: 
        quit("Empty image directory")

    for model in model_dir:
        if model_type not in model:
            print(model, model_type)
            model_dir.remove(model)


def print_profiling_data_pyarmmn_and_return_times(profiler):

    profiler_data = ann.get_profiling_data(profiler)
    
    times = profiler_data.inference_data["execution_time"]
    tot_time = 0
    for time in times:
        print(f"inference model time: {round(time/1000, 5)}ms")
        tot_time += time

    avg_time = tot_time / len(times)
 
    print(f"Total_time: {round(tot_time/1000,5)}ms, avg_time: {round(avg_time/1000, 5)}ms")
    return [time/1000 for time in times]

def write_profiling_data_pyarmnn(profiler, model_path, csv_path):
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

def tflite_runtime(model_dir, img_dir, label_dir, niter, img_result_dir, img_result_mask_dir, colormap):
    #source: 
    #https://www.tensorflow.org/lite/guide/inference
    #https://github.com/NXPmicro/pyarmnn-release/tree/master/python/pyarmnn/examples
    print("tflite")

    results = []
    inf_times = []


    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_dir)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']



    for i in range(niter):
        for img in img_dir:
            img_result_file = os.path.join(img_result_dir, img.split("/")[-1])
            img_result_mask_file = os.path.join(img_result_mask_dir, img.split("/")[-1])
            img_org = cv2.imread(img)
            img, image_data = preprocess_image_tflite_deeplabv3(img, input_shape[1], input_shape[2], input_type)

            interpreter.set_tensor(input_details[0]['index'], image_data)

            beg = time()
            interpreter.invoke()
            end = time()
            inf_time = end-beg
            inf_times.append(inf_time*1000)
            print(inf_time*1000)



            results.append(output_data_tflite_runtime(output_details, interpreter, img, img_org, img_result_file, img_result_mask_file, label_dir, colormap))
            #sys.exit("Finish")

            #results.append(output_data_tflite_runtime(output_details, interpreter, img_org, img_result_file, label_dir))
    
    return results
    

def pyarmnn(model_dir, img_dir, label_dir, niter, csv_path, img_result_dir, img_result_mask_dir, colormap):

    print("pyarmnn")
    # LINK TO CODE: https://www.youtube.com/watch?v=HQYosuy4ABY&t=1867s
    #https://developer.arm.com/documentation/102557/latest
    #file:///C:/Users/Maroun_Desktop_PC/SynologyDrive/Bachelorarbeit/pyarmnn/pyarmnn_doc.html#pyarmnn.IOutputSlot

    results = []

    print(f"Working with ARMNN {ann.ARMNN_VERSION}")

    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(model_dir)

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
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

    input_type = np.float32

    if ann.TensorInfo.IsQuantized(input_tensor_info):
        print(np.uint8)
        input_type = np.uint8
    
    #inference 
    results = []
    for i in range(niter):
        for img in img_dir:
            img_result_file = os.path.join(img_result_dir, img.split("/")[-1])
            img_result_mask_file = os.path.join(img_result_mask_dir, img.split("/")[-1])
            img_org = cv2.imread(img)

            image, image_data = preprocess_image_tflite_deeplabv3(img, height, width, input_type)

            input_tensors = ann.make_input_tensors([input_binding_info], [image_data])
            runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
            result = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
            results.append(output_data_pyarmnn(result, image, img_org, img_result_file, img_result_mask_file, label_dir, colormap))

    write_profiling_data_pyarmnn(profiler, model_dir, csv_path)

    return results

def openvino(model_dir, img_dir, label_dir, niter, img_result_dir, img_result_mask_dir, colormap):

    results = []
    print("openvino")
    print(model_dir)

    device_name = "CPU"

    # --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

# --------------------------- Step 2. Read a model --------------------------------------------------------------------
    log.info(f'Reading the model: {model_dir}')
    # (.xml and .bin files) or (.onnx file)
    model = core.read_model(model_dir)

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1

    if len(model.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

    for i in range(niter):
        for img in img_dir:
            img_result_file = os.path.join(img_result_dir, img.split("/")[-1])
            img_result_mask_file = os.path.join(img_result_mask_dir, img.split("/")[-1])
            img_org = cv2.imread(img)
            log.info(f'Reading the model: {model_dir}')
            # (.xml and .bin files) or (.onnx file)
            model = core.read_model(model_dir)

            if len(model.inputs) != 1:
                log.error('Sample supports only single input topologies')
                return -1

            if len(model.outputs) != 1:
                log.error('Sample supports only single output topologies')
                return -1

    # --------------------------- Step 3. Set up input --------------------------------------------------------------------
            # Read input image
            image = cv2.imread(img)
            print(image.shape)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Add N dimension
            input_tensor = np.expand_dims(image, 0)
            print("Input shape", input_tensor.shape)


            # Preprpocess
            input_tensor, model = preprocess_image_ov_deeplabv3(input_tensor, model)


    # --------------------------- Step 5. Loading model to the device -----------------------------------------------------
            log.info('Loading the model to the plugin')
            compiled_model = core.compile_model(model, device_name)

    # --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
            log.info('Starting inference in synchronous mode')
            start_time = time()
            result = compiled_model.infer_new_request({0: input_tensor})
            end_time = time()
            print(end_time-start_time)


    # --------------------------- Step 7. Process output ------------------------------------------------------------------
            #predictions = next(iter(result.values()))
            #probs = predictions.reshape(-1)
            results.append(output_data_ov(result, image, img_org, img_result_file, img_result_mask_file, label_dir, colormap))

    return results

def onnx_runtime(model_dir, img_dir_list, label_dir, niter, json_path, img_result_dir, img_result_mask_dir, colormap):

    print("label_dir", label_dir)
                 
    results = []

    options = onnxruntime.SessionOptions()
    options.enable_profiling = True

    session = onnxruntime.InferenceSession(model_dir, options)

    input_name = session.get_inputs()[0].name
    print(session.get_outputs()[0].name)
    outputs = [session.get_outputs()[0].name]
    print(outputs)
    #output_name = session.get_outputs()[0].name
    #print(output_name)

    #outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]

    image_height = session.get_inputs()[0].shape[1]
    image_width = session.get_inputs()[0].shape[2]
    
    print("------------------------------------------")
    print(image_height, image_width)
    print("-----------------------------------------")

    input_data_type = session.get_inputs()[0].type
    output_data_type = session.get_outputs()[0].type

    for i in range(niter):
        for img in img_dir_list:
            img_result_file = os.path.join(img_result_dir, img.split("/")[-1])
            img_result_mask_file = os.path.join(img_result_mask_dir, img.split("/")[-1])
            img_org = cv2.imread(img)
            output = session.run(outputs, {input_name:preprocess_image_onnx_deeplabv3(img, input_data_type, image_height, image_width)})
            #print("Output: ", output.shape)
            #print("Output 0: ", output[0].shape)




            #output = session.run([output_name], {input_name: img_data})[0]
            #quit()

            #output = output.flatten()
            #output = softmax(output) # this is optional
            results.append(output_data_onnx(output, img_org, img_org, img_result_file, img_result_mask_file, label_dir, colormap))
        
    prof_file = session.end_profiling()
    print(prof_file)

    os.rename(prof_file, os.path.join(json_path, prof_file))
      
    return results


def pytorch(model_dir, img_dir_list, label_dir, niter, json_path, img_result_dir, img_result_mask_dir, colormap):
    print("Pytorch")

    results = []

    model = torch.hub.load('pytorch/vision:v0.10.0', model_dir, pretrained=True)
    model.eval()

    preprocess = preprocess_image_pytorch_deeplabv3()

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):

            for i in range(niter):
                for img in img_dir_list:
                    img_result_file = os.path.join(img_result_dir, img.split("/")[-1])
                    img_result_mask_file = os.path.join(img_result_mask_dir, img.split("/")[-1])
                    input_image = Image.open(img)
                    input_image = input_image.convert("RGB")

                    #img = 'https://ultralytics.com/images/zidane.jpg'
                    img_org = img_org = cv2.imread(img)

                    input_tensor = preprocess(input_image)
                    input_batch = input_tensor.unsqueeze(0)

                    start_time = time()
                    with torch.no_grad():
                        output = model(input_batch)['out'][0]
                    end_time = time()
                    print(end_time-start_time)
                    print(output)


                    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
                    #probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    #get_result(label_dir, probabilities)
                    #output_predictions = output.argmax(0)

                    results.append(output_data_pytorch_deeplabv3(output, img_org, img_org, img_result_file, img_result_mask_file, label_dir, colormap))

    prof.export_chrome_trace(os.path.join(json_path, model_dir))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    return results
    

def handle_model_dir(args):

    model_dir_list = []

    if args.pytorch_model_name and args.api == "pytorch":
        model_dir_list.append(args.pytorch_model_name)
    else:
        if args.model:
            model_dir_list.append(args.model)
        elif args.model_folder:
            models = os.listdir(args.model_folder)

            for model in models:
                if "._" not in model:
                    model_dir_list.append(os.path.join(args.model_folder, model))
        else:
            quit("No model or model folder given")

    return model_dir_list

def handle_img_dir(args):
    image_dir_list = []

    print("Image path: ", args.image)
    print("Image path: ", args.image_folder)

    if args.image:
        image_dir_list.append(args.image)
    elif args.image_folder:
        images = os.listdir(args.image_folder)

        for img in images:
            if ".jpg" in img and "._" not in img:
                image_dir_list.append(os.path.join(args.image_folder, img))
    else:
        quit("No img or image folder given")

    print("Image list: ", image_dir_list)
    return image_dir_list


def handle_label_dir(args):
    if args.labels:
        return args.labels
    else:
        quit("No labels folder specified")


def handle_arguments():
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module for Classification')

    parser.add_argument("-api", '--api', help='inference API', required=False)

    parser.add_argument("-m", '--model', help='model path', required=False)
    parser.add_argument("-mf", "--model_folder", help="model_folder_path", required=False)

    parser.add_argument("-img", "--image", help="path of a picture", required=False)
    parser.add_argument("-imgf", "--image_folder", help="image folder path", required=False)

    parser.add_argument("-l", "--labels", help="txt file with classes", required=False)

    parser.add_argument("-o", "--output", help="where the results are saved", required=False, default="/home/pi/sambashare/BacArbeit/results/segmentation/")

    parser.add_argument("-s", '--sleep', default=1,type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-n", '--niter', default=1, type=int, help='number of iterations', required=False)
    parser.add_argument("-cm", "--colormap", required=False)
    parser.add_argument("-url", "--pytorch_model_name", help="gives the name of the pytorch model which has to be downloaded from the internet", required=False)

    return parser.parse_args()

def build_dir_paths(args):
    general_dir = os.path.abspath(os.path.dirname(__file__)).split("scripts")[0]

    model_dir_list = handle_model_dir(args=args)
    img_dir_list = handle_img_dir(args=args)
    label_dir = handle_label_dir(args=args)
    inf_times_dir = os.path.join(args.output, "inference_time")
    result_dir = os.path.join(args.output, "prediction")
    img_result_dir = os.path.join(args.output, "images")
    img_result_mask_dir = os.path.join(args.output, "images_mask")


    return general_dir, model_dir_list, img_dir_list, label_dir, inf_times_dir, result_dir, img_result_dir, img_result_mask_dir

def handle_other_args_par(args):
    sleep = args.sleep
    niter = args.niter

    if args.colormap == "ade20k":
        #print(args.dataset)
        colormap = create_ade20k_label_colormap()
    elif args.colormap == "pascal_voc_2012":
        colormap = create_pascal_label_colormap()
    elif args.colormap == "cityscapes":
        colormap = create_cityscapes_label_colormap()
    else:
        sys.exit("No vaild name for dataset given")

    return sleep, niter, colormap
    

def main():
    
    profiler = cProfile.Profile()

    args = handle_arguments()
    general_dir, model_dir_list, img_dir_list, label_dir, inf_times_dir, result_dir, img_result_dir, img_result_mask_dir = build_dir_paths(args=args)
    sleep, niter, colormap = handle_other_args_par(args=args)



    print(model_dir_list)

    if args.api == "tflite_runtime":
        check_directories(model_dir_list, img_dir_list, ".tflite")
        
        for model in model_dir_list:
            model_name = model.split("/")[-1].split(".tflite")[0] + "_tflite_runtime.txt"
            inf_times_file = os.path.join(inf_times_dir, model_name)
            result_file = os.path.join(result_dir, model_name)
            profiler.enable()
            results = tflite_runtime(model, img_dir_list, label_dir, niter, img_result_dir, img_result_mask_dir, colormap)
            profiler.disable()
            
            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")

            with open(inf_times_file, 'w') as stream:
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats()
            
     
    elif args.api == "pyarmnn":
        check_directories(model_dir_list, img_dir_list, ".tflite")

        for model in model_dir_list:
            csv_path = os.path.join(args.output, "pyarmnn_profiler")
            model_name_txt = model.split("/")[-1].split(".tflite")[0] + "_pyarmnn.txt"
            model_name_csv = model.split("/")[-1].split(".tflite")[0] + "_pyarmnn.csv"
            inf_times_file = os.path.join(inf_times_dir, model_name_txt)
            result_file = os.path.join(result_dir, model_name_txt)
            csv_path = os.path.join(csv_path, model_name_csv)

            profiler.enable()
            results = pyarmnn(model, img_dir_list, label_dir, niter, csv_path, img_result_dir, img_result_mask_dir, colormap)
            profiler.disable()

            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")

            with open(inf_times_file, 'w') as stream:
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats()

    elif args.api == "onnx":
        check_directories(model_dir_list, img_dir_list, ".onnx")

        for model in model_dir_list:
            json_path = os.path.join(args.output, "onnx_profiler")
            model_name_txt = model.split("/")[-1].split(".onnx")[0] + "_onnx.txt"
            inf_times_file = os.path.join(inf_times_dir, model_name_txt)
            result_file = os.path.join(result_dir, model_name_txt)

            profiler.enable()
            results = onnx_runtime(model, img_dir_list, label_dir, niter, json_path, img_result_dir, img_result_mask_dir, colormap)
            profiler.disable()

            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")

            with open(inf_times_file, 'w') as stream:
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats()

    elif args.api == "pytorch":
        for model in model_dir_list:
            json_path = os.path.join(args.output, "pytorch_profiler")
            if args.pytorch_model_name:
                model_name_txt = args.pytorch_model_name + "_pytorch.txt"
            else:
                model_name_txt = model.split("/")[-1].split(".pth")[0] + "_pytorch.txt"
            inf_times_file = os.path.join(inf_times_dir, model_name_txt)
            result_file = os.path.join(result_dir, model_name_txt) 

            profiler.enable()
            results = pytorch(model, img_dir_list, label_dir, niter, json_path, img_result_dir, img_result_mask_dir, colormap)
            profiler.disable()

            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")

            with open(inf_times_file, 'w') as stream:
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats()
    
    elif args.api == "ov":
        check_directories(model_dir_list, img_dir_list, ".xml")
    
        for model in model_dir_list:
            model_name_txt = model.split("/")[-1].split(".xml")[0] + "_ov.txt"
            inf_times_file = os.path.join(inf_times_dir, model_name_txt)
            result_file = os.path.join(result_dir, model_name_txt) 

            profiler.enable()
            results = openvino(model, img_dir_list, label_dir, niter, img_result_dir, img_result_mask_dir, colormap)
            profiler.disable()

            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")

            with open(inf_times_file, 'w') as stream:
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats()



if __name__ == "__main__":
    main()

