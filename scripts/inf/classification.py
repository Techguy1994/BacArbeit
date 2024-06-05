import time 
import lib.postprocess as post
import lib.preprocess as pre
import lib.helper as helper
from time import perf_counter
import lib.data as dat
import pandas as pd
import time
import numpy as np
from PIL import Image
import cv2
import os 
import sys

def run_tf(args):
    #import tensorflow as tf

    try:
        import tflite_runtime.interpreter as tflite

        #delegate_input
        if args.api == "delegate":
            print("armnn tflite delegate")

            if os.path.exists("/home/pi/sambashare/armnn-24.02/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so"):
                libarmnnDelegate = "/home/pi/sambashare/armnn-24.02/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so"
                print(os.path.exists("/home/pi/sambashare/armnn-24.02/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so"))
            elif os.path.exists("/home/pi/sambashare/armnn-24.02/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so"):
                print(os.path.exists("/home/pi/sambashare/armnn-24.02/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so"))
            #sys.exit()
            
            #/home/pi/sambashare/armnn-24.02/build-tool/scripts/aarch64_build/delegate
            #armnn_delegate = tflite.load_delegate(library=libarmnnDelegate,
            #                                options={"backends": "CpuAcc,CpuRef", "logging-severity":"info", "number-of-threads": args.num_threads})
            
            armnn_delegate = tflite.load_delegate(library=libarmnnDelegate,
                                options={"backends": "CpuAcc", "number-of-threads": args.num_threads, "reduce-fp32-to-fp16": True, "enable-fast-math": True, "logging-severity":"info", "disable-tflite-runtime-fallback": True})
            
            interpreter = tflite.Interpreter(model_path=args.model, experimental_delegates=[armnn_delegate], num_threads=args.num_threads)
        else:
            interpreter = tflite.Interpreter(model_path=args.model, experimental_delegates=None, num_threads=args.num_threads)
    except:
        import tensorflow as tf

        if args.api == "delegate":
            print("delegate")
            armnn_delegate = tf.lite.experimental.load_delegate(library="/home/pi/sambashare/armnn-24.02/build-tool/scripts/aarch64_build/delegate/libarmnnDelegate.so",
                                            options={"backends": "CpuAcc,CpuRef", "logging-severity":"info"})
            interpreter = tf.lite.Interpreter(model_path=args.model, experimental_delegates=[armnn_delegate], num_threads=args.num_threads)
        else:
            interpreter = tf.lite.Interpreter(model_path=args.model, experimental_delegates=None, num_threads=args.num_threads)

    output_dict = dat.create_base_dictionary_class(args.n_big)

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']

    for i in range(args.niter):
        for image in args.images:
            processed_image = pre.preprocess_tflite_moobilenet(image, input_shape[1], input_shape[2], input_type)
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
                output_data = interpreter.get_tensor(output_details[0]['index'])
                output = post.handle_output_tf(output_data, output_details, args.label, args.n_big)
                output_dict = dat.store_output_dictionary_class(output_dict, image, lat, output, args.n_big)
            else: 
                output_dict = dat.store_output_dictionary_only_lat(output_dict, image, lat, args.n_big)

            df = dat.create_pandas_dataframe(output_dict)

        time.sleep(args.sleep)

    return df

def run_pyarmnn(args):
    import pyarmnn as ann
    

    output_dict = dat.create_base_dictionary_class(args.n_big)

    print(f"Working with ARMNN {ann.ARMNN_VERSION}")

    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(args.model)

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)
    print(f"{runtime.GetDeviceSpec()}\n")


    #Optimziation Options
    preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
    opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    #print(f"Preferred Backends: {preferredBackends}\n")
    print(f"Optimizationon warnings: {messages}")
    

    # get input binding information for the input layer of the model
    graph_id = parser.GetSubgraphCount() - 1
    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
    input_tensor_id = input_binding_info[0]
    input_tensor_info = input_binding_info[1]
    height, width = input_tensor_info.GetShape()[1], input_tensor_info.GetShape()[2]
    print(f"tensor id: {input_tensor_id},tensor info: {input_tensor_info}\n")

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
            processed_image = pre.preprocess_tflite_moobilenet(image, height, width, data_type)
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
                result = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
                output = post.handle_output_pyarmnn(result, args.label, args.n_big)
                output_dict = dat.store_output_dictionary_class(output_dict, image, lat, output, args.n_big)
            else: 
                output_dict = dat.store_output_dictionary_only_lat(output_dict, image, lat, args.n_big)
            df = dat.create_pandas_dataframe(output_dict)

        time.sleep(args.sleep)

    return df

def run_onnx(args):
    import onnxruntime

    output_dict = dat.create_base_dictionary_class(args.n_big)

    options = onnxruntime.SessionOptions()

    #, 'XNNPACKExecutionProvider'
    print(args.num_threads)
    providers = ['CPUExecutionProvider']
    options.intra_op_num_threads = args.num_threads
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    #options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    #options.inter_op_num_threads = 8
    #macht keinen Unterschied in meinen Tests (MobilenetV2)
    #options.add_session_config_entry('session.dynamic_block_base', '8') 
    #options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = onnxruntime.InferenceSession(args.model, options, providers=providers)
    print(session.get_providers())

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image_height = session.get_inputs()[0].shape[2]
    image_width = session.get_inputs()[0].shape[3]

    input_data_type = session.get_inputs()[0].type
    output_data_type = session.get_outputs()[0].type

    for i in range(args.niter):
        for image in args.images:
            #print(image)
            processed_image = pre.preprocess_onnx_mobilenet(image, image_height, image_width, input_data_type)

            if args.profiler == "perfcounter":
                start_time = perf_counter()
                result = session.run([output_name], {input_name:processed_image})[0]
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                lat = 0
                result = session.run([output_name], {input_name:processed_image})[0]
                
            if not args.skip_output: 
                output = post.handle_output_onnx_mobilenet_class(result, output_data_type, args.label, args.n_big)
                output_dict = dat.store_output_dictionary_class(output_dict, image, lat, output, args.n_big)
            else: 
                output_dict = dat.store_output_dictionary_only_lat(output_dict, image, lat, args.n_big)
            df = dat.create_pandas_dataframe(output_dict)


        time.sleep(args.sleep)

    return df

def run_pytorch(args):

    import torch
    from torchvision import models, transforms
    import sys
    import lib.load_pytorch_models as pt
    
    output_dict = dat.create_base_dictionary_class(args.n_big)

    
    print(args.num_threads)
    torch.set_num_threads(args.num_threads)

    if args.model == "mobilenet_v2":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    elif args.model == "mobilenet_v3_large":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_large', pretrained=True)
    elif args.model == "mobilenet_v3_small":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=True)
    elif args.model == "mobilenet_v3_large_q":
        torch.backends.quantized.engine = 'qnnpack'
        model = models.quantization.mobilenet_v3_large(pretrained=True, quantize=True)
    elif args.model == "mobilenet_v2_q":
        torch.backends.quantized.engine = 'qnnpack'
        model = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
    #func_call = pt.load_pytorch_model(args.model)

    #func_call = "models." + args.model + "(pretrained=True)"
    #model = func_call
    #print(func_call)
    #model = eval(func_call)

    #model = models.mobilenet_v2(pretrained=True)
        
    model = torch.jit.script(model)

    model.eval()

    preprocess = pre.preprocess_pytorch_mobilenet()

    for i in range(args.niter):
        for image in args.images:
            input_image = Image.open(image)
            #print(len(np.shape(np.asarray(input_image).astype(float))))

            if len(np.shape(np.asarray(input_image).astype(np.float32))) == 2:
                input_image = input_image.convert("RGB")

            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0) 

            if args.profiler == "perfcounter":
                start_time = perf_counter()
                with torch.no_grad():
                    output = model(input_batch)
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
            else:
                output = model(input_batch)

            if not args.skip_output:
                output = post.handle_output_pytorch_mobilenet_class(output, args.label, args.n_big)
                output_dict = dat.store_output_dictionary_class(output_dict, image, lat, output, args.n_big)
            else:
                output_dict = dat.store_output_dictionary_only_lat(output_dict, image, lat, args.n_big)
            df = dat.create_pandas_dataframe(output_dict)
            

        time.sleep(args.sleep)

    return df

def run_sync_ov(args):

    from openvino.runtime import InferRequest, AsyncInferQueue
    from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
    from openvino import Core, Layout, Type

    import logging as log
    import sys

    print("Chosen API: Sync Openvino")
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    output_dict = dat.create_base_dictionary_class(args.n_big)

    device_name = "CPU"

        # --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

    #print(args.model)

    #if "FP32" in args.model:
    #    float32 = True
    #elif "FP16" in args.model
    #sys.exit()

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
    print(model.input().shape)
    #h, w = 224, 224

    #resized_images = [cv2.resize(image, (w, h)) for image in images]

    # Add N dimension
    #input_tensors = [np.expand_dims(image, 0) for image in resized_images]

    # --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
    ppp = PrePostProcessor(model)

    # 1) Set input tensor information:
    # - input() provides information about a single model input
    # - precision of tensor is supposed to be 'u8'
    # - layout of data is 'NHWC'
    ppp.input().tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC'))  # noqa: N400
    
    # - apply linear resize from tensor spatial dims to model spatial dims
    ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)

    # 2) Here we suppose model has 'NCHW' layout for input
    #ppp.input().model().set_layout(Layout('NCHW'))
    ppp.input().model().set_layout(Layout('NHWC'))

    # 3) Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    ppp.output().tensor().set_element_type(Type.f32)

    # 4) Apply preprocessing modifing the original 'model'
    model = ppp.build()

    print(str(args.num_threads))

    # --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    log.info('Loading the model to the plugin')
    config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": str(args.num_threads)} #"NUM_STREAMS": "1"} 
    #config = {"PERFORMANCE_HINT": "THROUGHPUT", "INFERENCE_NUM_THREADS": str(args.num_threads), "NUM_STREAMS": "4"} 
    compiled_model = core.compile_model(model, device_name, config)
    #compiled_model = core.compile_model(model, device_name)
    num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    print("optimal number of requests", num_requests)

    for i in range(args.niter):
        for image in args.images:
            img_org = cv2.imread(image)
            input_tensor = pre.preprocess_ov_mobilenet(shape, img_org)

            if args.profiler == "perfcounter":
                start_time = perf_counter()
                result = compiled_model.infer_new_request({0: input_tensor})
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
                
                df = dat.create_pandas_dataframe(output_dict)
            else:
                result = compiled_model.infer_new_request({0: input_tensor})

            if not args.skip_output:
                output = post.handle_output_openvino_moiblenet_class(result, args.label, args.n_big)
                #print(output)
                output_dict = dat.store_output_dictionary_class(output_dict, image, lat, output, args.n_big)
            else:
                output_dict = dat.store_output_dictionary_only_lat(output_dict, image, lat, args.n_big)

            df = dat.create_pandas_dataframe(output_dict)

        time.sleep(args.sleep)

    return df


def run_async_ov(args):
    print("todo")





