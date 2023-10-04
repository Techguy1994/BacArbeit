import time 
import lib.postprocess as post
import lib.preprocess as pre
from time import perf_counter
import lib.data as dat
import pandas as pd
import os 
import cv2
import numpy as np
from PIL import Image

def run_tf(args, output_image_folder):
    
    import tensorflow as tf


    output_dict = dat.create_base_dictionary_det()
    
    interpreter = tf.lite.Interpreter(model_path=args.model, experimental_delegates=None, num_threads=4)

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']

    for i in range(args.niter):
        for image in args.images:
            image_result_file = os.path.join(output_image_folder, image.split("/")[-1])
            processed_image, unprocessed_image =pre.preprocess_tflite_yolov5(image, input_shape[1], input_shape[2], input_type)
            interpreter.set_tensor(input_details[0]['index'], processed_image)

            if args.profiler == "perfcounter":
                start_time = perf_counter()
                interpreter.invoke()
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
                output = post.handle_output_tf_yolo_det(output_details, interpreter, unprocessed_image, args.thres, image_result_file, args.label)
                output_dict = dat.store_output_dictionary_det(output_dict, image, lat, output)
                df = dat.create_pandas_dataframe(output_dict)
                print("pandas output: ", df)

        time.sleep(args.sleep)     

    return df 

def run_pyarmnn(args, output_image_folder):

    import pyarmnn as ann
    import csv

    output_dict = dat.create_base_dictionary_det()

    print(f"Working with ARMNN {ann.ARMNN_VERSION}")

    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(args.model)

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)
    print(f"{runtime.GetDeviceSpec()}\n")

    preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
    opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    net_id, _ = runtime.LoadNetwork(opt_network)

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

    if ann.TensorInfo.IsQuantized(input_tensor_info):
        input_type = np.uint8
    else:
        input_type = np.float32

    for i in range(args.niter):
        for image in args.images:
            image_result_file = os.path.join(output_image_folder, image.split("/")[-1])
            processed_image, unprocessed_image =pre.preprocess_tflite_yolov5(image, width, height, input_type)

            input_tensors = ann.make_input_tensors([input_binding_info], [processed_image])
            if args.profiler:
                start_time = perf_counter()
                runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)

                output = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
                output = post.handle_output_pyarmnn_yolo_det(output, unprocessed_image, args.thres, image_result_file, args.label)
                output_dict = dat.store_output_dictionary_det(output_dict, image, lat, output)
                df = dat.create_pandas_dataframe(output_dict)
                print("pandas output: ", df)

        time.sleep(args.sleep)  

    return df


def run_onnx(args, output_image_folder):
    print("Chosen API: Onnx runtime")

    import onnxruntime
    import json

    output_dict = dat.create_base_dictionary_det()

    options = onnxruntime.SessionOptions()
    providers = ['CPUExecutionProvider']

    session = onnxruntime.InferenceSession(args.model, options, providers=providers)
    print(session.get_providers())

    input_name = session.get_inputs()[0].name
    outputs = []
    #print(session.get_outputs()[0].name, session.get_outputs()[1].name, session.get_outputs()[2].name, session.get_outputs()[3].name)
    #outputs = [session.get_outputs()[0].name, session.get_outputs()[1].name, session.get_outputs()[2].name, session.get_outputs()[3].name]
    for ses in session.get_outputs():
        outputs.append(ses.name)
    #outputs = [session.get_outputs()[0].name]
    print(outputs)
    #output_name = session.get_outputs()[0].name
    #print(output_name)

    #outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]

    image_height = session.get_inputs()[0].shape[2]
    image_width = session.get_inputs()[0].shape[3]

    input_data_type = session.get_inputs()[0].type
    output_data_type = session.get_outputs()[0].type

    for i in range(args.niter):
        for image in args.images:
            image_result_file = os.path.join(output_image_folder, image.split("/")[-1])
            img_org = cv2.imread(image)

            if args.profiler == "perfcounter":
                start_time = perf_counter()
                output = session.run(outputs, {input_name: pre.preprocess_onnx_yolov5(image, input_data_type, image_height, image_width)})
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)

                output = post.handle_output_onnx_yolo_det(output, img_org, args.thres, image_result_file, args.label,(image_height, image_width))
                output_dict = dat.store_output_dictionary_det(output_dict, image, lat, output)
                df = dat.create_pandas_dataframe(output_dict)
                print("pandas output: ", df)

        time.sleep(args.sleep)     

    return df

def run_pytorch(args, output_image_folder):

    import torch
    from torchvision import models, transforms

    output_dict = dat.create_base_dictionary_det()

    model = torch.hub.load("ultralytics/yolov5", "yolov5l")

    model.eval()

    preprocess = pre.preprocess_pytorch_yolo()

    for i in range(args.niter):
        for image in args.images:
            image_result_file = os.path.join(output_image_folder, image.split("/")[-1])
            img_org = cv2.imread(image)
            input_image = Image.open(image)

            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0) 

            if args.profiler == "perfcounter":
                start_time = perf_counter()
                with torch.no_grad():
                    output = model(input_batch)
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)

                output = post.handle_output_pytorch_yolo_det(output, img_org, args.thres, image_result_file, args.label,(1, 1))
                output_dict = dat.store_output_dictionary_det(output_dict, image, lat, output)
                df = dat.create_pandas_dataframe(output_dict)
                print("pandas output: ", df)

        time.sleep(args.sleep)     

    return df



