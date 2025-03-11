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

def run_tf(args, raw_folder, overlay_folder, index_folder):

    #import tflite_runtime.interpreter as tflite
    print("Chosen API: tflite runtime intepreter")

    output_dict = dat.create_base_dictionary_seg()

    results = []
    inf_times = []

    try:
        import tflite_runtime.interpreter as tflite
        print("tensorflow-runtime")

        #delegate_input
        if args.api == "delegate":
            print("armnn tflite delegate")
            print(args.num_threads)

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
            raw_file = image.split("/")[-1]
            raw_file = raw_file.split(".")[0] + ".png"
            raw_file = os.path.join(raw_folder, raw_file)

            overlay_file = os.path.join(overlay_folder, image.split("/")[-1])

            index_file = image.split("/")[-1]
            index_file = index_file.split(".")[0] + ".png"
            index_file = os.path.join(index_folder, index_file)

            if args.profiler == "perfcounter":
                preprocessed_image = pre.preprocess_tf_deeplab(image, input_shape, input_type)
                interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

                start_time = perf_counter()
                interpreter.invoke()
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                lat = 0
                interpreter.invoke()

            if not args.skip_output:
                output = post.handle_output_deeplab_tf(output_details, interpreter, original_image, raw_file, overlay_file, index_file, args.colormap, args.label)
                output_dict = dat.store_output_dictionary_seg(output_dict, image, lat, output)
            else:
                output_dict = dat.store_output_dictionary_seg_only_lat(output_dict, image, lat)
                
        time.sleep(args.sleep)  

    df = dat.create_pandas_dataframe(output_dict)  

    return df

def run_pyarmnn(args, raw_folder, overlay_folder, index_folder):
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

    input_shape = (0, width, height)


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
            raw_file = image.split("/")[-1]
            raw_file = raw_file.split(".")[0] + ".png"
            raw_file = os.path.join(raw_folder, raw_file)

            overlay_file = os.path.join(overlay_folder, image.split("/")[-1])

            index_file = image.split("/")[-1]
            index_file = index_file.split(".")[0] + ".png"
            index_file = os.path.join(index_folder, index_file)


            processed_image = pre.preprocess_tf_deeplab(image, input_shape, data_type)
            input_tensors = ann.make_input_tensors([input_binding_info], [processed_image])

            if args.profiler == "perfcounter":

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
                output = post.handle_output_deeplab_pyarmnn(output, original_image, raw_file, overlay_file, index_file, args.colormap, args.label)
                output_dict = dat.store_output_dictionary_seg(output_dict, image, lat, output)
            else:
                output_dict = dat.store_output_dictionary_seg_only_lat(output_dict, image, lat)
                
        time.sleep(args.sleep)     

    
    df = dat.create_pandas_dataframe(output_dict)
    print(df)
    return df


def run_onnx(args, raw_folder, overlay_folder, index_folder):
    print("Chosen API: Onnx runtime")

    import onnxruntime

    output_dict = dat.create_base_dictionary_seg()

    options = onnxruntime.SessionOptions()
    if args.profiler == "onnx":
        options.enable_profiling = True

    # 'XNNPACKExecutionProvider'
    providers = ['CPUExecutionProvider']

    if args.num_threads:
        print("set thread")
        options.intra_op_num_threads = args.num_threads

    session = onnxruntime.InferenceSession(args.model, options, providers=providers)
    print(session.get_providers())

    input_name = session.get_inputs()[0].name
    #print(session.get_outputs()[0].name)
    outputs = [session.get_outputs()[0].name]
    #print(outputs)
    #output_name = session.get_outputs()[0].name
    #print(output_name)

    #outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]

    print("input shape", session.get_inputs()[0].shape[1], session.get_inputs()[0].shape[2], session.get_inputs()[0].shape[3])
    if session.get_inputs()[0].shape[1] == 3:
        image_width = session.get_inputs()[0].shape[2]
        image_height = session.get_inputs()[0].shape[3]
    elif session.get_inputs()[0].shape[3] == 3:
        image_width = session.get_inputs()[0].shape[1]
        image_height = session.get_inputs()[0].shape[2]

    input_data_type = session.get_inputs()[0].type
    output_data_type = session.get_outputs()[0].type

    for i in range(args.niter):
        for image in args.images:
            original_image = cv2.imread(image)
            raw_file = image.split("/")[-1]
            raw_file = raw_file.split(".")[0] + ".png"
            raw_file = os.path.join(raw_folder, raw_file)

            overlay_file = os.path.join(overlay_folder, image.split("/")[-1])

            index_file = image.split("/")[-1]
            index_file = index_file.split(".")[0] + ".png"
            index_file = os.path.join(index_folder, index_file)

            processed_image = pre.preprocess_onnx_deeplab_alt(image , input_data_type, image_height, image_width)
            print(processed_image.shape)

            if args.profiler == "perfcounter":

                start_time = perf_counter()
                output = session.run(outputs, {input_name:processed_image})[0]
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                lat = 0
                output = session.run(outputs, {input_name:processed_image})[0]

            if not args.skip_output:
                output = post.handle_output_deeplab_onnx_alt(output, original_image, raw_file, overlay_file, index_file, args.colormap, args.label)
                output_dict = dat.store_output_dictionary_seg(output_dict, image, lat, output)
            else:
                output_dict = dat.store_output_dictionary_seg_only_lat(output_dict, image, lat)
                

                
        time.sleep(args.sleep)     

    df = dat.create_pandas_dataframe(output_dict)

    if args.profiler == "onnx":
        return df, session
    else:
        return df
                
                

def run_pytorch(args, raw_folder, overlay_folder, index_folder):
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
        print(args.model)
        sys.exit("Nothing found")
    
    print(args.num_threads)
    if args.num_threads:
        print("set thread")
        torch.set_num_threads(args.num_threads)

    preprocess = pre.preprocess_pytorch_seg()

    model.eval()

    for i in range(args.niter):
        for image in args.images:
            original_image = cv2.imread(image)
            raw_file = image.split("/")[-1]
            raw_file = raw_file.split(".")[0] + ".png"
            raw_file = os.path.join(raw_folder, raw_file)

            overlay_file = os.path.join(overlay_folder, image.split("/")[-1])

            index_file = image.split("/")[-1]
            index_file = index_file.split(".")[0] + ".png"
            index_file = os.path.join(index_folder, index_file)

            input_batch = pre.preprocess_pytorch_deeplab(image, preprocess)

            if args.profiler == "perfcounter":
                
                start_time = perf_counter()
                with torch.no_grad():
                    output = model(input_batch)['out'][0]
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                with torch.no_grad():
                    output = model(input_batch)['out'][0]

            if not args.skip_output:
                output = post.handle_output_deeplab_pytorch(output, original_image, raw_file, overlay_file, index_file, args.colormap, args.label)
                output_dict = dat.store_output_dictionary_seg(output_dict, image, lat, output)
            else:
                output_dict = dat.store_output_dictionary_seg_only_lat(output_dict, image, lat)
                

                
        time.sleep(args.sleep)     

    df = dat.create_pandas_dataframe(output_dict)
    #print(df)
    return df

def run_sync_openvino(args, raw_folder, overlay_folder, index_folder):
    from openvino.runtime import InferRequest, AsyncInferQueue
    from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat
    from openvino import Core, Layout, Type

    import logging as log
    import sys

    print("Chosen API: Sync Openvino")
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    output_dict = dat.create_base_dictionary_seg()

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
    images = [cv2.imread(image_path) for image_path in args.images]
    _, h, w, _ = model.input().shape
    #print(model.input().shape)
    resized_images = [cv2.resize(img, (w,h)) for img in images]
    bgr_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in resized_images]
    input_tensors = [np.expand_dims(img, 0) for img in bgr_images]
    print("input tensor shape: ", input_tensors[0].shape)
    
    # --------------------------- Step 3. Set up input --------------------------------------------------------------------
    # Read input images
    #images = [cv2.imread(image_path) for image_path in args.images]

    # Resize images to model input dims
    #_, _, h, w = model.input().shape
    #_, h, w, _ = model.input().shape
    #print("Model input shape: ",model.input().shape)
    #h, w = 224, 224

    #resized_images = [cv2.resize(image, (513, 513)) for image in images]

    # Add N dimension
    #input_tensors = [np.expand_dims(image, 0) for image in resized_images]
    

    # --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
    # Step 4. Inizialize Preprocessing for the model
    #ppp = PrePostProcessor(model)
    # Specify input image format
    #ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.RGB)
    #  Specify preprocess pipeline to input image without resizing
    #ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB).scale([255., 255., 255.])
    # Specify model's input layout
    #ppp.input().model().set_layout(Layout("NHWC"))
    #  Specify output results format
    #ppp.output().tensor().set_element_type(Type.f32)
    # Embed above steps in the graph
    #model = ppp.build()
    #compiled_model = core.compile_model(model, "CPU")

    # --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
    print("Preprocess")
    ppp = PrePostProcessor(model)
    
    ppp.input().tensor() \
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

# --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    #log.info('Loading the model to the plugin')
    #compiled_model = core.compile_model(model, device_name)

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
        for j, input_tensor in enumerate(input_tensors):
            raw_file = args.images[j].split("/")[-1]
            print(raw_file)
            raw_file = raw_file.split(".")[0] + ".png"
            raw_file = os.path.join(raw_folder, raw_file)

            overlay_file = os.path.join(overlay_folder, args.images[j].split("/")[-1])

            index_file = args.images[j].split("/")[-1]
            index_file = index_file.split(".")[0] + ".png"
            index_file = os.path.join(index_folder, index_file)
            
            img_org = cv2.imread(args.images[j])
            print(args.images[j])
            #image_height, image_width = img_org.shape[1], img_org.shape[0]
            
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

                output = post.handle_output_deeplab_ov(result, img_org, raw_file, overlay_file, index_file, args.colormap, args.label)
                output_dict = dat.store_output_dictionary_seg(output_dict, args.images[j], lat, output)
            else:
                output_dict = dat.store_output_dictionary_seg_only_lat(output_dict, args.images[j], lat)

        time.sleep(args.sleep) 


    df = dat.create_pandas_dataframe(output_dict)
    print("pandas output: ", df)    

    return df
    

def run_pytorch_with_profiler(args, raw_folder, overlay_folder, index_folder):
    import torch
    from torchvision import models, transforms
    from torch.profiler import profile, record_function, ProfilerActivity

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
        print(args.model)
        sys.exit("Nothing found")

    preprocess = pre.preprocess_pytorch_seg()

    model.eval()

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):

            for i in range(args.niter):
                for image in args.images:
                    original_image = cv2.imread(image)
                    raw_file = image.split("/")[-1]
                    raw_file = raw_file.split(".")[0] + ".png"
                    raw_file = os.path.join(raw_folder, raw_file)

                    overlay_file = os.path.join(overlay_folder, image.split("/")[-1])

                    index_file = image.split("/")[-1]
                    index_file = index_file.split(".")[0] + ".png"
                    index_file = os.path.join(index_folder, index_file)

                    input_batch = pre.preprocess_pytorch_deeplab(image, preprocess)

                    if args.profiler == "perfcounter":
                        
                        start_time = perf_counter()
                        with torch.no_grad():
                            output = model(input_batch)['out'][0]
                        end_time = perf_counter()
                        lat = end_time - start_time
                        print("time in ms: ", lat*1000)
                    else:
                        with torch.no_grad():
                            output = model(input_batch)['out'][0]

                    if not args.skip_output:
                        output = post.handle_output_deeplab_pytorch(output, original_image, raw_file, overlay_file, index_file, args.colormap, args.label)
                        output_dict = dat.store_output_dictionary_seg(output_dict, image, lat, output)
                    else:
                        output_dict = dat.store_output_dictionary_seg_only_lat(output_dict, image, lat)
                

                
        time.sleep(args.sleep)     

    df = dat.create_pandas_dataframe(output_dict)
    print(df)
    return df