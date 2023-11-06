import time 
import lib.postprocess as post
import lib.preprocess as pre
from time import perf_counter
import lib.data as dat
import pandas as pd
import cv2
import os 
import sys   
import numpy as np
from PIL import Image

def run_tf(args, raw_folder, overlay_folder):

    import tflite_runtime.interpreter as tflite
    print("Chosen API: tflite runtime intepreter")

    output_dict = dat.create_base_dictionary_seg()

    results = []
    inf_times = []

    #delegate_input
    if args.api == "delegate":
        armnn_delegate = tflite.load_delegate(library="/home/pi/sambashare/armnn_bld/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so",
                                          options={"backends": "CpuAcc,CpuRef", "logging-severity":"info"})
        interpreter = tflite.Interpreter(model_path=args.model, experimental_delegates=[armnn_delegate], num_threads=4)
    else:
        interpreter = tflite.Interpreter(model_path=args.model, experimental_delegates=None, num_threads=4)

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']

    for i in range(args.niter):
        for image in args.images:
            original_image = cv2.imread(image)
            raw_file = os.path.join(raw_folder, image.split("/")[-1])
            overlay_file = os.path.join(overlay_folder, image.split("/")[-1])

            if args.profiler == "perfcounter":
                image, processed_image = pre.preprocess_tf_deeplab(image, input_shape[1], input_shape[2], input_type)
                interpreter.set_tensor(input_details[0]['index'], processed_image)

                start_time = perf_counter()
                interpreter.invoke()
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                lat = 0
                interpreter.invoke()

            if not args.skip_output:
                output = post.handle_output_deeplab_tf(output, interpreter, original_image, raw_file, overlay_file, args.colormap, args.label)
                output_dict = dat.store_output_dictionary_seg(output_dict, image, lat, output)
            else:
                output_dict = dat.store_output_dictionary_seg_only_lat(output_dict, image, lat)
                
        time.sleep(args.sleep)   

    df = dat.create_pandas_dataframe(output_dict)  

    return df

def run_pyarmnn(args, raw_folder, overlay_folder):
    import pyarmnn as ann
    import csv

    output_dict = dat.create_base_dictionary_seg()

    print(f"Working with ARMNN {ann.ARMNN_VERSION}")

    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(args.model)

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)
    print(f"{runtime.GetDeviceSpec()}\n")

    preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
    opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

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

    if ann.TensorInfo.IsQuantized(input_tensor_info):
        data_type = np.uint8
    else:
        data_type= np.float32

    for i in range(args.niter):
        for image in args.images:
            original_image = cv2.imread(image)
            raw_file = os.path.join(raw_folder, image.split("/")[-1])
            overlay_file = os.path.join(overlay_folder, image.split("/")[-1])

            if args.profiler == "perfcounter":
                image, processed_image = pre.preprocess_tf_deeplab(image, height, width, data_type)
                input_tensors = ann.make_input_tensors([input_binding_info], [processed_image])

                start_time = perf_counter()
                runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                lat = 0
                runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call

            if not args.skip_output:
                output = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
                output = post.handle_output_deeplab_pyarmnn(output, original_image, raw_file, overlay_file, args.colormap, args.label)
                output_dict = dat.store_output_dictionary_seg(output_dict, image, lat, output)
            else:
                output_dict = dat.store_output_dictionary_seg_only_lat(output_dict, image, lat)
                
        time.sleep(args.sleep)     

    
    df = dat.create_pandas_dataframe(output_dict)
    print(df)
    return df


def run_onnx(args, raw_folder, overlay_folder):
    print("Chosen API: Onnx runtime")

    import onnxruntime

    output_dict = dat.create_base_dictionary_seg()

    options = onnxruntime.SessionOptions()

    # 'XNNPACKExecutionProvider'
    providers = ['CPUExecutionProvider']

    session = onnxruntime.InferenceSession(args.model, options, providers=providers)
    print(session.get_providers())

    input_name = session.get_inputs()[0].name
    #print(session.get_outputs()[0].name)
    outputs = [session.get_outputs()[0].name]
    #print(outputs)
    #output_name = session.get_outputs()[0].name
    #print(output_name)

    #outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]

    image_width = session.get_inputs()[0].shape[1]
    image_height = session.get_inputs()[0].shape[2]

    input_data_type = session.get_inputs()[0].type
    output_data_type = session.get_outputs()[0].type

    for i in range(args.niter):
        for image in args.images:
            original_image = cv2.imread(image)
            raw_file = os.path.join(raw_folder, image.split("/")[-1])
            overlay_file = os.path.join(overlay_folder, image.split("/")[-1])

            if args.profiler == "perfcounter":
                processed_image = pre.preprocess_onnx_deeplab(original_image, input_data_type, image_height, image_width)

                start_time = perf_counter()
                output = session.run(outputs, {input_name:processed_image})[0]
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                lat = 0
                output = session.run(outputs, {input_name:processed_image})[0]

            if not args.skip_output:
                output = post.handle_output_deeplab_onnx(output, original_image, raw_file, overlay_file, args.colormap, args.label)
                output_dict = dat.store_output_dictionary_seg(output_dict, image, lat, output)
            else:
                output_dict = dat.store_output_dictionary_seg_only_lat(output_dict, image, lat)
                

                
        time.sleep(args.sleep)     

    df = dat.create_pandas_dataframe(output_dict)
    print(df)
    return df
                
                

def run_pytorch(args, raw_folder, overlay_folder):
    import torch
    from torchvision import models, transforms

    output_dict = dat.create_base_dictionary_seg()

    if args.model == "deeplabv3_resnet50":
        model = torch.hub.load('pytorch/vision:v0.10.0',"deeplabv3_resnet50", pretrained=True)
    elif args.model == "deeplabv3_resnet101":
        model = torch.hub.load('pytorch/vision:v0.10.0',"deeplabv3_resnet101", pretrained=True)
    elif args.model == "deeplabv3_mobilenet_v3_large":
        model = torch.hub.load('pytorch/vision:v0.10.0',"deeplabv3_mobilenet_v3_large", pretrained=True)
    elif args.model == "deeplabv3_mobilenet_v3_small":
        model = torch.hub.load('pytorch/vision:v0.10.0',"deeplabv3_mobilenet_v3_small", pretrained=True)
    else: 
        sys.exit("Nothing found")

    preprocess = pre.preprocess_pytorch_seg()

    model.eval()

    for i in range(args.niter):
        for image in args.images:
            original_image = cv2.imread(image)
            raw_file = os.path.join(raw_folder, image.split("/")[-1])
            overlay_file = os.path.join(overlay_folder, image.split("/")[-1])

            if args.profiler == "perfcounter":
                input_batch = pre.preprocess_pytorch_deeplab(image, preprocess)

                start_time = perf_counter()
                with torch.no_grad():
                    output = model(input_batch)['out'][0]
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                output = post.handle_output_deeplab_pytorch(output, original_image, raw_file, overlay_file, args.colormap, args.label)
                output_dict = dat.store_output_dictionary_seg(output_dict, image, lat, output)
                

                
        time.sleep(args.sleep)     

    df = dat.create_pandas_dataframe(output_dict)
    print(df)
    return df

def run_sync_openvino():
    print("todo")

#def run_async_openvino():
#    print("todo")