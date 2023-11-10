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
    
    import tflite_runtime.interpreter as tflite

    output_dict = dat.create_base_dictionary_det()
    
    #delegate_input
    if args.api == "delegate":
        print("delegate")
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
            image_result_file = os.path.join(output_image_folder, image.split("/")[-1])
            processed_image, unprocessed_image =pre.preprocess_tflite_yolov5(image, input_shape[1], input_shape[2], input_type)
            interpreter.set_tensor(input_details[0]['index'], processed_image)

            if args.profiler == "perfcounter":
                start_time = perf_counter()
                interpreter.invoke()
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                interpreter.invoke()
                lat = 0

            if not args.skip_output:
                output = post.handle_output_tf_yolo_det(output_details, interpreter, unprocessed_image, args.thres, image_result_file, args.label)
                output_dict = dat.store_output_dictionary_det(output_dict, image, lat, output)
            else: 
                output_dict = dat.store_output_dictionary_det_only_lat(output_dict, image, lat)


        time.sleep(args.sleep)     

    df = dat.create_pandas_dataframe(output_dict)
    print("pandas output: ", df)

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
            else: 
                lat = 0
                runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
            
            if not args.skip_output:     
                output = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
                output = post.handle_output_pyarmnn_yolo_det(output, unprocessed_image, args.thres, image_result_file, args.label)
                output_dict = dat.store_output_dictionary_det(output_dict, image, lat, output)
            else:
                output_dict = dat.store_output_dictionary_det_only_lat(output_dict, image, lat)
        time.sleep(args.sleep)  

    df = dat.create_pandas_dataframe(output_dict)
    print("pandas output: ", df)

        

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
            else:
                lat = 0
                output = session.run(outputs, {input_name: pre.preprocess_onnx_yolov5(image, input_data_type, image_height, image_width)})
            
            if not args.skip_output:
                output = post.handle_output_onnx_yolo_det(output, img_org, args.thres, image_result_file, args.label,(image_height, image_width))
                output_dict = dat.store_output_dictionary_det(output_dict, image, lat, output)
            else:
                output_dict = dat.store_output_dictionary_det_only_lat(output_dict, image, lat)

            df = dat.create_pandas_dataframe(output_dict)
            print("pandas output: ", df)

        time.sleep(args.sleep) 

    df = dat.create_pandas_dataframe(output_dict)
    print("pandas output: ", df)    

    return df

def run_pytorch(args, output_image_folder):

    import torch
    from torchvision import models, transforms

    output_dict = dat.create_base_dictionary_det()

    model = torch.hub.load("ultralytics/yolov5", "yolov5l", pretrained=True)

    model.eval()

    preprocess = pre.preprocess_pytorch_yolo()

    print(args.images)

    for i in range(args.niter):
        for image in args.images:
            image_result_file = os.path.join(output_image_folder, image.split("/")[-1])
            img_org = cv2.imread(image)
            input_image = Image.open(image)

            input_tensor = preprocess(input_image)
            print(input_tensor.shape)
            #input_batch = input_tensor
            input_batch = input_tensor.unsqueeze(0) 
            print(input_batch.shape)

            if args.profiler == "perfcounter":
                start_time = perf_counter()
                with torch.no_grad():
                    output = model(image)
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                lat = 0
                with torch.no_grad():
                    output = model(input_image)

            if not args.skip_output:
                output = post.handle_output_pytorch_yolo_det(output, img_org, args.thres, image_result_file, args.label,(1, 1))
                output_dict = dat.store_output_dictionary_det(output_dict, image, lat, output)
            else:
                output_dict = dat.store_output_dictionary_det_only_lat(output_dict, image, lat)

        time.sleep(args.sleep)    

    df = dat.create_pandas_dataframe(output_dict)
    print("pandas output: ", df) 

    return df

def run_sync_ov(args, output_image_folder):
    from openvino.runtime import InferRequest, AsyncInferQueue
    from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat
    from openvino import Core, Layout, Type

    import logging as log
    import sys

    print("Chosen API: Sync Openvino")
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    output_dict = dat.create_base_dictionary_det()

    device_name = "CPU"

    # --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

# --------------------------- Step 2. Read a model --------------------------------------------------------------------
    log.info(f'Reading the model: {args.model}')
    # (.xml and .bin files) or (.onnx file)
    model = core.read_model(args.model)

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1

    if len(model.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1
    
    # --------------------------- Step 3. Set up input --------------------------------------------------------------------
    # Read input images
    images = [cv2.imread(image_path) for image_path in args.images]

    # Resize images to model input dims
    _, _, h, w = model.input().shape
    #_, h, w, _ = model.input().shape
    print("Model input shape: ",model.input().shape)
    #h, w = 224, 224

    resized_images = [cv2.resize(image, (640, 640)) for image in images]

    

    # Add N dimension
    input_tensors = [np.expand_dims(image, 0) for image in resized_images]
    print("input tensor shape: ", input_tensors[0].shape)

    # --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
    # Step 4. Inizialize Preprocessing for the model
    ppp = PrePostProcessor(model)
    # Specify input image format
    ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.BGR)
    #  Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB).scale([255., 255., 255.])
    # Specify model's input layout
    ppp.input().model().set_layout(Layout("NCHW"))
    #  Specify output results format
    ppp.output().tensor().set_element_type(Type.f32)
    # Embed above steps in the graph
    model = ppp.build()
    compiled_model = core.compile_model(model, "CPU")

    # --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    log.info('Loading the model to the plugin')
    config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": "4", "NUM_STREAMS": "4"} #"PERFORMANCE_HINT_NUM_REQUESTS": "1"} findet nicht
    compiled_model = core.compile_model(model, device_name, config)
    #compiled_model = core.compile_model(model, device_name)
    num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    print("optimal number of requests", num_requests)


    for i in range(args.niter):
        for j, input_tensor in enumerate(input_tensors):
            image_result_file = os.path.join(output_image_folder, args.images[j].split("/")[-1])
            img_org = cv2.imread(args.images[j])
            print(args.images[j])
            image_height, image_width = img_org.shape[1], img_org.shape[0]

            if args.profiler == "perfcounter":
                start_time = perf_counter()
                result = compiled_model.infer_new_request({0: input_tensor})
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                lat = 0
                result = compiled_model.infer_new_request({0: input_tensor})
            
            if not args.skip_output:
                print(result)

                output = post.handle_output_ov_yolo_det(result, img_org, args.thres, image_result_file, args.label,(640, 640))
                output_dict = dat.store_output_dictionary_det(output_dict, args.images[j], lat, output)
            else:
                output_dict = dat.store_output_dictionary_det_only_lat(output_dict, args.images[j], lat)

        time.sleep(args.sleep) 

    df = dat.create_pandas_dataframe(output_dict)
    print("pandas output: ", df)    

    return df


