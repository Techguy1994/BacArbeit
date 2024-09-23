import time 
import lib.postprocess as post
import lib.preprocess as pre
from time import perf_counter
import lib.data as dat
import lib.directories as d
import pandas as pd
import os 
import cv2
import numpy as np
from PIL import Image
import sys

def run_tf(args, output_image_folder):

    print(args.num_threads)

    try:
        import tflite_runtime.interpreter as tflite
        print("tensorflow-runtime")

        #delegate_input
        if args.api == "delegate":
            print("armnn tflite delegate")
            print("hey")

            if os.path.exists("/home/pi/sambashare/armnn/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so"):
                libarmnnDelegate = "/home/pi/sambashare/armnn/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so"
                print(os.path.exists("/home/pi/sambashare/armnn/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so"))
            #elif os.path.exists("/home/pi/sambashare/armnn/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so"):
            #    print(os.path.exists("/home/pi/sambashare/armnn/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so"))
            #sys.exit()
            else:
                print("delegate not found")
                sys.exit()
            
            #/home/pi/sambashare/armnn-24.02/build-tool/scripts/aarch64_build/delegate
            #armnn_delegate = tflite.load_delegate(library=libarmnnDelegate,
            #                                options={"backends": "CpuAcc,CpuRef", "logging-severity":"info", "number-of-threads": args.num_threads})
            
            armnn_delegate = tflite.load_delegate(library=libarmnnDelegate,
                                options={"backends": "CpuAcc, CpuRef", "number-of-threads": args.num_threads, "reduce-fp32-to-fp16": True, "enable-fast-math": True})
            if args.num_threads:
                interpreter = tflite.Interpreter(model_path=args.model, experimental_delegates=[armnn_delegate], num_threads=args.num_threads)
            else:
                interpreter = tflite.Interpreter(model_path=args.model, experimental_delegates=[armnn_delegate])
        else:
            if args.num_threads:
                interpreter = tflite.Interpreter(model_path=args.model, experimental_delegates=None, num_threads=args.num_threads)
            else:
                interpreter = tflite.Interpreter(model_path=args.model, experimental_delegates=None)
    except:
        import tensorflow as tf
        print("tensorflow")

        if args.api == "delegate":
            print("delegate")
            armnn_delegate = tf.lite.experimental.load_delegate(library="/home/pi/sambashare/armnn-24.02/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so",
                                            options={"backends": "CpuAcc,CpuRef", "logging-severity":"info"})
            if args.num_threads:
                print(args.num_threads)
                interpreter = tf.lite.Interpreter(model_path=args.model, experimental_delegates=[armnn_delegate], num_threads=args.num_threads)
            else:
                interpreter = tf.lite.Interpreter(model_path=args.model, experimental_delegates=[armnn_delegate])
        else:
            if args.num_threads:
                print(args.num_threads)
                interpreter = tf.lite.Interpreter(model_path=args.model, experimental_delegates=None, num_threads=args.num_threads)
            else: 
                interpreter = tf.lite.Interpreter(model_path=args.model, experimental_delegates=None) 

    output_dict = dat.create_base_dictionary_det()

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
    #print("pandas output: ", df)

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

    output_dict = dat.create_base_dictionary_det()

    options = onnxruntime.SessionOptions()
    if args.profiler == "onnx":
        options.enable_profiling = True
    providers = ['CPUExecutionProvider']

    if args.num_threads:
        print("set thread")
        options.intra_op_num_threads = args.num_threads
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    session = onnxruntime.InferenceSession(args.model, options, providers=providers)
    #print(session.get_providers())

    input_name = session.get_inputs()[0].name
    outputs = []
    #print(session.get_outputs()[0].name, session.get_outputs()[1].name, session.get_outputs()[2].name, session.get_outputs()[3].name)

    for ses in session.get_outputs():
        outputs.append(ses.name)
    #outputs = [session.get_outputs()[0].name]
    #print(outputs)
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
                output = session.run(outputs, {input_name: pre.preprocess_onnx_yolov5(img_org, input_data_type, image_height, image_width)})
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                lat = 0
                output = session.run(outputs, {input_name: pre.preprocess_onnx_yolov5(img_org, input_data_type, image_height, image_width)})
            
            if not args.skip_output:
                output = post.handle_output_onnx_yolo_det(output, img_org, args.thres, image_result_file, args.label,(image_height, image_width))
                output_dict = dat.store_output_dictionary_det(output_dict, image, lat, output)
            else:
                output_dict = dat.store_output_dictionary_det_only_lat(output_dict, image, lat)

            df = dat.create_pandas_dataframe(output_dict)
            #print("pandas output: ", df)

        time.sleep(args.sleep) 

    df = dat.create_pandas_dataframe(output_dict)
    #print("pandas output: ", df)    

    if args.profiler == "onnx":
        return df, session
    else:
        return df

def run_pytorch(args, output_image_folder):

    import torch
    from torchvision import models, transforms

    output_dict = dat.create_base_dictionary_det()

    print(args.num_threads)
    if args.num_threads:
        print("set thread")
        torch.set_num_threads(args.num_threads)

    if args.model == "yolov5l":
        model = torch.hub.load("ultralytics/yolov5", "yolov5l", pretrained=True)
    elif args.model == "yolov5m":
        model = torch.hub.load("ultralytics/yolov5", "yolov5m", pretrained=True)
    elif args.model == "yolov5n":
        model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
    elif args.model == "yolov5s":
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    elif "yolov7" in args.model:
        model = torch.hub.load('WongKinYiu/yolov7', 'custom', args.model, force_reload=True, trust_repo=True)
    elif "yolov3" in args.model:
        model = torch.hub.load("ultralytics/yolov3", "custom", args.model, force_reload=True, trust_repo=True)
    elif "yolov8" in args.model:
        from ultralytics import YOLO
        print("yolov8")
        model = YOLO(args.model)
        
    

    model.eval()

    preprocess = pre.preprocess_pytorch_yolo()

    #print(args.images)

    for i in range(args.niter):
        for image in args.images:
            image_result_file = os.path.join(output_image_folder, image.split("/")[-1])
            img_org = cv2.imread(image)
            input_image = Image.open(image)

            #input_tensor = preprocess(input_image)
            #print(input_tensor.shape)
            #input_batch = input_tensor
            #input_batch = input_tensor.unsqueeze(0) 
            #print(input_batch.shape)

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
    #print("pandas output: ", df) 

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
    #images = [cv2.imread(image_path) for image_path in args.images]

    # Resize images to model input dims
    shape = model.input().shape
    #_, h, w, _ = model.input().shape
    print("Model input shape: ",model.input().shape)
    #h, w = 224, 224

    #resized_images = [cv2.resize(image, (w, h)) for image in images]

    

    # Add N dimension
    #input_tensors = [np.expand_dims(image, 0) for image in resized_images]
    #print("input tensor shape: ", input_tensors[0].shape)
    

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
    if args.num_threads:
        print("set thread")
        config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": str(args.num_threads)} #"NUM_STREAMS": "1"} 
    else: 
        config = {"PERFORMANCE_HINT": "LATENCY"} 
    compiled_model = core.compile_model(model, device_name, config)
    #compiled_model = core.compile_model(model, device_name)
    num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    print("optimal number of requests", num_requests)

 
    for i in range(args.niter):
        for image in args.images:
            
            
            image_result_file = os.path.join(output_image_folder, image.split("/")[-1])
            img_org = cv2.imread(image)
            
            image_height, image_width = img_org.shape[1], img_org.shape[0]
            input_tensor = pre.preprocess_ov_yolo(shape, img_org)


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

                output = post.handle_output_ov_yolo_det(result, img_org, args.thres, image_result_file, args.label,(640, 640))
                output_dict = dat.store_output_dictionary_det(output_dict, image, lat, output)
            else:
                output_dict = dat.store_output_dictionary_det_only_lat(output_dict, image, lat)

        time.sleep(args.sleep) 

    df = dat.create_pandas_dataframe(output_dict)
    #print("pandas output: ", df)    

    return df

def run_pytorch_with_profiler(args, output_image_folder):

    import torch
    from torchvision import models, transforms
    from torch.profiler import profile, record_function, ProfilerActivity

    output_dict = dat.create_base_dictionary_det()

    torch.set_num_threads(args.num_threads)

    if args.model == "yolov5l":
        model = torch.hub.load("ultralytics/yolov5", "yolov5l", pretrained=True)
    elif args.model == "yolov5m":
        model = torch.hub.load("ultralytics/yolov5", "yolov5m", pretrained=True)
    elif args.model == "yolov5n":
        model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
    elif args.model == "yolov5s":
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    elif "yolov7" in args.model:
        model = torch.hub.load('WongKinYiu/yolov7', 'custom', args.model, force_reload=True, trust_repo=True)
    elif "yolov3" in args.model:
        model = torch.hub.load("ultralytics/yolov3", "custom", args.model, force_reload=True, trust_repo=True)
    elif "yolov8" in args.model:
        from ultralytics import YOLO
        print("yolov8")
        model = YOLO(args.model)
        
    

    model.eval()

    preprocess = pre.preprocess_pytorch_yolo()

    #print(args.images)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):

            for i in range(args.niter):
                for image in args.images:
                    image_result_file = os.path.join(output_image_folder, image.split("/")[-1])
                    img_org = cv2.imread(image)
                    input_image = Image.open(image)

                    #input_tensor = preprocess(input_image)
                    #print(input_tensor.shape)
                    #input_batch = input_tensor
                    #input_batch = input_tensor.unsqueeze(0) 
                    #print(input_batch.shape)

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
    #print("pandas output: ", df) 

    return df

