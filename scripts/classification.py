import argparse
import logging as log
import os
import sys
from time import sleep, perf_counter
import cProfile, pstats
import numpy as np
import cv2
from PIL import Image

InferRequest = None


def return_n_biggest_result_pytorch(output_data, class_dir, n_big=10):

    with open(class_dir, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    results = []

    probabilities = torch.nn.functional.softmax(output_data[0], dim=0)

    #prob = probabilities.item()
    #print(torch.sum(probabilities))

    max_positions = np.argpartition(probabilities, -n_big)[-n_big:]


    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = probabilities[entry] 

        results.append({"object": categories[entry], "Index:": entry.item(), "Accuracy": val.item()})
        
    return results

def return_n_biggest_result_ov(result, class_dir, n_big=10):

    results = []

    with open(class_dir, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    predictions = next(iter(result.values()))
    probs = predictions.reshape(-1)

    max_positions = np.argpartition(probs, -n_big)[-n_big:]
    out_normalization_factor = 1

    #print(output_details[0]["dtype"])

    #if "integer" in output_details:
    #    print("int")
    #    quit("no adapted to onnx, please change following code when quantized model is given")
    #    out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    #elif "float" in output_details:
    #    print("float")
    #    out_normalization_factor = 1

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = probs[entry] / out_normalization_factor
        #result[entry] = [val*100]
        #print("\tpos {} : {:.2f}%".format(entry, val*100))
        results.append({"object": categories[entry], "Index:": entry, "Accuracy": val})
        
    return results

def return_n_biggest_result_pyarmnn(output_data, class_dir, n_big=3):

    with open(class_dir, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    results = []

    output_data = output_data[0]
    max_positions = np.argpartition(output_data[0], -n_big)[-n_big:]

    if output_data.dtype == "uint8":
        out_normalization_factor = np.iinfo(output_data.dtype).max
    elif output_data.dtype == "float32":
        out_normalization_factor = 1
    
    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[0][entry] / out_normalization_factor
        results.append({"object": categories[entry], "Index:": entry, "Accuracy": val})
        
    return results

def return_n_biggest_result_tflite_runtime(output_data, output_details, class_dir, n_big=3):

    results = []

    with open(class_dir, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    max_positions = np.argpartition(output_data[0], -n_big)[-n_big:]
    #print(output_details[0]["dtype"])

    if output_details[0]['dtype'] == np.uint8:
        out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    elif output_details[0]['dtype'] == np.float32:
        out_normalization_factor = 1

    

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[0][entry] / out_normalization_factor
        results.append({"object":  categories[entry], "Index": entry, "Accuracy": val})

    return results

def return_n_biggest_result_onnx(output_data, output_details, class_dir, n_big=10):

    results = []

    with open(class_dir, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    output_data = output_data.flatten()
    output_data = softmax(output_data) # this is optional

    max_positions = np.argpartition(output_data, -n_big)[-n_big:]
    out_normalization_factor = 1

    #print(output_details[0]["dtype"])

    if "integer" in output_details:
        print("int")
        quit("not adapted to onnx, please change following code when quantized model is given")
        out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    elif "float" in output_details:
        #print("float")
        out_normalization_factor = 1


    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[entry] / out_normalization_factor

        results.append({"object":  categories[entry], "Index": entry, "Accuracy": val})
        
    return results

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess_image_pytorch_mobilenet3():

    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    return preprocess

def preprocess_image_ov_mobilenet2(input_tensor, model):
    # --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
        ppp = PrePostProcessor(model)

        _, h, w, _ = input_tensor.shape

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
        ppp.input().model().set_layout(Layout('NCHW'))

        # 4) Set output tensor information:
        # - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(Type.f32)

        # 5) Apply preprocessing modifying the original 'model'
        model = ppp.build()

        return input_tensor, model

def preprocess_image_onnx_mobilenet3(image_path, height, width, data_type):

    if "float" in data_type:
        type = np.float32
    else:
        type = np.uint8

    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS)
    image_data = np.asarray(image).astype(type)
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = image_data[channel, :, :] / 255 - mean[channel] / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

def preprocess_image_tflite_mobilenet3(image_path, height, width, data_type):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS)
    image_data = np.asarray(image).astype(data_type)
    if data_type is np.float32:
        for channel in range(image_data.shape[0]):
            image_data[channel, :, :] = (image_data[channel, :, :] / 127.5) - 1
    image_data = np.expand_dims(image_data, 0)
    return image_data

def check_directories(model_dir, img_dir, model_type):

    if not model_dir:
        quit("Empty model directory")
    if not img_dir: 
        quit("Empty image directory")

    for model in model_dir:
        if model_type not in model:
            print(model, model_type)
            model_dir.remove(model)

def setup_profiling(net_id, runtime):
    profiler = runtime.GetProfiler(net_id)
    profiler.EnableProfiling(True)
    return profiler

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

def tflite_runtime(model_dir, img_dir, label_dir, n_big, niter, optimize):
    #source: 
    #https://www.tensorflow.org/lite/guide/inference
    #https://github.com/NXPmicro/pyarmnn-release/tree/master/python/pyarmnn/examples
    print("Chosen API: tflite runtime intepreter")

    results = []
    inf_times = []


    # Load the TFLite model and allocate tensors.

    if optimize:
        print("optimize")
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_dir, experimental_delegates=None, num_threads=2)
        #interpreter = tflite.Interpreter(model_path=model_dir, experimental_delegates=None, num_threads=2)
    else: 
        import tflite_runtime.interpreter as tflite
        #interpreter = tf.lite.Interpreter(model_path=model_dir)
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
            print("Image: ", img)
            img = preprocess_image_tflite_mobilenet3(img, input_shape[1], input_shape[2], input_type)

            interpreter.set_tensor(input_details[0]['index'], img)

            beg = perf_counter()
            interpreter.invoke()
            end = perf_counter()
            diff = end - beg
            print("time in ms: ", diff*1000)

            output_data = interpreter.get_tensor(output_details[0]['index'])

            results.append(return_n_biggest_result_tflite_runtime(output_data=output_data, output_details=output_details, n_big=n_big, class_dir=label_dir))
            inf_times.append(diff)
    
    return results, inf_times
    
def pyarmnn(model_dir, img_dir, label_dir, n_big, niter, csv_path, en_profiler):

    print("Chosen API: PyArmnn")
    # LINK TO CODE: https://www.youtube.com/watch?v=HQYosuy4ABY&t=1867s
    #https://developer.arm.com/documentation/102557/latest
    #file:///C:/Users/Maroun_Desktop_PC/SynologyDrive/Bachelorarbeit/pyarmnn/pyarmnn_doc.html#pyarmnn.IOutputSlot

    global ann, csv
    import pyarmnn as ann
    import csv

    results = []
    inf_times = []

    print(f"Working with ARMNN {ann.ARMNN_VERSION}")

    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(model_dir)

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

    # Setup the Profilier for layer and network and inference time 
    if en_profiler:
        profiler = setup_profiling(net_id, runtime)
    
    if ann.TensorInfo.IsQuantized(input_tensor_info):
        data_type = np.uint8
    else:
        data_type= np.float32

    for i in range(niter):
        for img in img_dir:

            print("Image: ", img)

            image = preprocess_image_tflite_mobilenet3(img, height, width, data_type)

            input_tensors = ann.make_input_tensors([input_binding_info], [image])
            beg = perf_counter()
            runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
            end = perf_counter()
            diff = end - beg
            print("Time in ms: ", diff*1000)
            result = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
            results.append(return_n_biggest_result_pyarmnn(result, label_dir, n_big))
            inf_times.append(diff)
        
    if en_profiler:
        write_profiling_data_pyarmnn(profiler, model_dir, csv_path)

    return results, inf_times

def openvino(model_dir, img_dir, label_dir, n_big, niter, optimize):


    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    global PrePostProcessor, ResizeAlgorithm, Core, Layout, Type

    from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
    from openvino.runtime import Core, Layout, Type

    results = []
    inf_times = []
    device_name = "CPU"
    

    # --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

    # --------------------------- Step 2. Read a model --------------------------------------------------------------------
    log.info(f'Reading the model: {model_dir}')
    # (.xml and .bin files) or (.onnx file)
    model = core.read_model(model_dir)

    caching_supported = 'EXPORT_IMPORT' in core.get_property(device_name, 'OPTIMIZATION_CAPABILITIES')
    print("Caching support? ", caching_supported)

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1

    if len(model.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

    for i in range(niter):
        for img in img_dir:
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
            # Add N dimension

            input_tensor = np.expand_dims(image, 0)

            # Preprpocess
            input_tensor, model = preprocess_image_ov_mobilenet2(input_tensor, model)


    # --------------------------- Step 5. Loading model to the device -----------------------------------------------------
            log.info('Loading the model to the plugin')
            #config = {"PERFORMANCE_HINT": "THROUGHPUT"}
            config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": "2", "NUM_STREAMS": "4"} #"PERFORMANCE_HINT_NUM_REQUESTS": "1"} findet nicht
            compiled_model = core.compile_model(model, device_name, config)
            num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
            print("optimal", num_requests)

    # --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
            log.info('Starting inference in synchronous mode')
            beg = perf_counter()
            result = compiled_model.infer_new_request({0: input_tensor})
            end = perf_counter()
            diff = end - beg
            print("Time: ", diff*1000)
            inf_times.append(diff)

    # --------------------------- Step 7. Process output ------------------------------------------------------------------

            results.append(return_n_biggest_result_ov(result, label_dir, n_big))

    return results, inf_times

def sync_openvino(model_dir, img_dir, label_dir, n_big, niter):
    print("Chosen API: Sync Openvino")
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    global AsyncInferQueue, Core, Layout, Type, PrePostProcessor, InferRequest

    from openvino.runtime import InferRequest
    from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
    from openvino.runtime import AsyncInferQueue, Core, Layout, Type

    results = []
    inf_times = []
    device_name = "CPU"

    print(img_dir)

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
    
    # --------------------------- Step 3. Set up input --------------------------------------------------------------------
    # Read input images
    images = [cv2.imread(image_path) for image_path in img_dir]

    # Resize images to model input dims
    _, _, h, w = model.input().shape
    #_, h, w, _ = model.input().shape
    print(model.input().shape)
    #h, w = 224, 224

    resized_images = [cv2.resize(image, (w, h)) for image in images]

    # Add N dimension
    input_tensors = [np.expand_dims(image, 0) for image in resized_images]

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
    ppp.input().model().set_layout(Layout('NCHW'))

    # 3) Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    ppp.output().tensor().set_element_type(Type.f32)

    # 4) Apply preprocessing modifing the original 'model'
    model = ppp.build()

    # --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    log.info('Loading the model to the plugin')
    config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": "4", "NUM_STREAMS": "4"} #"PERFORMANCE_HINT_NUM_REQUESTS": "1"} findet nicht
    compiled_model = core.compile_model(model, device_name, config)
    #compiled_model = core.compile_model(model, device_name)
    num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    print("optimal number of requests", num_requests)


    start_tot_time = perf_counter()

    # --------------------------- Step 7. Do inference --------------------------------------------------------------------
    for i in range(niter):
        for j, input_tensor in enumerate(input_tensors):
            beg = perf_counter()
            result = compiled_model.infer_new_request({0: input_tensor})
            end = perf_counter()
            diff = end - beg
            print("Time in ms:", diff*1000)
            results.append(return_n_biggest_result_ov(result, label_dir, n_big))
            inf_times.append(diff)

    end_tot_time = perf_counter()
    print((end_tot_time-start_tot_time)*1000)

    return results, inf_times

def async_openvino(model_dir, img_dir, label_dir, n_big, niter):
    print("Chosen API: Async Openvino")
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    global AsyncInferQueue, Core, Layout, Type, PrePostProcessor, InferRequest


    from openvino.runtime import InferRequest
    from openvino.preprocess import PrePostProcessor
    from openvino.runtime import AsyncInferQueue, Core, Layout, Type

    global class_dir, nbig
    
    class_dir = label_dir
    nbig = n_big


    results = []
    inf_times = []
    device_name = "CPU"

    print(img_dir)

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
    
    # --------------------------- Step 3. Set up input --------------------------------------------------------------------
    # Read input images
    images = [cv2.imread(image_path) for image_path in img_dir]

    # Resize images to model input dims
    _, _, h, w = model.input().shape
    #_, h, w, _ = model.input().shape
    print(model.input().shape)
    resized_images = [cv2.resize(image, (w, h)) for image in images]

    # Add N dimension
    input_tensors = [np.expand_dims(image, 0) for image in resized_images]

    # --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
    ppp = PrePostProcessor(model)

    # 1) Set input tensor information:
    # - input() provides information about a single model input
    # - precision of tensor is supposed to be 'u8'
    # - layout of data is 'NHWC'
    ppp.input().tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC'))  # noqa: N400

    # 2) Here we suppose model has 'NCHW' layout for input
    ppp.input().model().set_layout(Layout('NCHW'))

    # 3) Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    ppp.output().tensor().set_element_type(Type.f32)

    # 4) Apply preprocessing modifing the original 'model'
    model = ppp.build()

    # --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    log.info('Loading the model to the plugin')
    #config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": "2", "NUM_STREAMS": "4"}
    #compiled_model = core.compile_model(model, device_name, config)
    compiled_model = core.compile_model(model, device_name)
    num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    print("optimal number of requests", num_requests)

    # --------------------------- Step 6. Create infer request queue ------------------------------------------------------
    log.info('Starting inference in asynchronous mode')
    # create async queue with optimal number of infer requests
    infer_queue = AsyncInferQueue(compiled_model)
    infer_queue.set_callback(completion_callback)

    start_tot_time = perf_counter()

# --------------------------- Step 7. Do inference --------------------------------------------------------------------
    for i in range(niter):
        for j, input_tensor in enumerate(input_tensors):
            beg = perf_counter()
            infer_queue.start_async({0: input_tensor}, img_dir[j])
            end = perf_counter()
            diff = end - beg
            print("Time in ms:", diff*1000)
            #results.append(None)
            inf_times.append(diff)

    infer_queue.wait_all()
# ----------------------------------------------------------------------------------------------------------------------
    end_tot_time = perf_counter()
    print((end_tot_time-start_tot_time)*1000)



    return results, inf_times

def completion_callback(infer_request: InferRequest, image_path: str):

    results = []

    with open(class_dir, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    predictions = next(iter(infer_request.results.values()))

    probs = predictions.reshape(-1)

    max_positions = np.argpartition(probs, -nbig)[-nbig:]
    #print(max_positions)
    #out_normalization_factor = 1

    #print(output_details[0]["dtype"])

    #if "integer" in output_details:
    #    print("int")
    #    quit("no adapted to onnx, please change following code when quantized model is given")
    #    out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    #elif "float" in output_details:
    #    print("float")
    #    out_normalization_factor = 1

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = probs[entry] 
        #result[entry] = [val*100]
        #print("\tpos {} : {:.2f}%".format(entry, val*100))
        results.append({"object": categories[entry], "Index:": entry, "Accuracy": val})

    with open("async_ov.txt", "a") as file:
        file.writelines(str(results))
        file.writelines("\n\n")
        
    #print("res: ", return_results)
    #return results

def onnx_runtime(model_dir, img_dir_list, label_dir, n_big, niter, json_path, optimize, en_profiler):

    print("Chosen API: Onnx runtime")

    import onnxruntime
    import json
                 
    results = []
    inf_times = []

    options = onnxruntime.SessionOptions()

    if en_profiler:
        options.enable_profiling = True


    #, 'XNNPACKExecutionProvider'
    providers = ['CPUExecutionProvider']

    if optimize:
        print("optimize")
        options.intra_op_num_threads = 4
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        #options.inter_op_num_threads = 4
        #macht keinen Unterschied in meinen Tests (MobilenetV2)
        #options.add_session_config_entry('session.dynamic_block_base', '8') 
        #options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        

    session = onnxruntime.InferenceSession(model_dir, options, providers=providers)
    print(session.get_providers())

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image_height = session.get_inputs()[0].shape[2]
    image_width = session.get_inputs()[0].shape[3]

    input_data_type = session.get_inputs()[0].type
    output_data_type = session.get_outputs()[0].type

    for i in range(niter):
        for img in img_dir_list:
            input = preprocess_image_onnx_mobilenet3(img, image_height, image_width, input_data_type)
            beg = perf_counter()
            output = session.run([output_name], {input_name:input})[0]
            end = perf_counter()
            diff = end - beg
            print("Time in ms: ", diff*1000)

            results.append(return_n_biggest_result_onnx(output, output_data_type, label_dir, n_big))
            inf_times.append(diff)
        
    if en_profiler:
        prof_file = session.end_profiling()
        print(prof_file)
        os.replace(prof_file, os.path.join(json_path, prof_file))
      
    return results, inf_times

def pytorch(model_dir, img_dir_list, label_dir, n_big, niter, json_path, optimize, en_profiler, quantized):

    print("Chosen API: PyTorch")

    global models, transforms, torch

    if en_profiler:
        from torch.profiler import profile, record_function, ProfilerActivity
    
    import torch
    from torchvision import models, transforms


    if optimize:
        print("Optimize")
        torch.set_num_threads(4)
        torch.backends.quantized.engine = 'qnnpack'

    #print(models.list_models())


    results = []
    inf_times = []


    #print(models.list_models())
    if quantized:
        func_call = "models.quantization." + model_dir + "(pretrained=True, quantize=True)"
        model = eval(func_call)

    else:
        func_call = "models." + model_dir + "(pretrained=True)"
        model = eval(func_call)


    preprocess = preprocess_image_pytorch_mobilenet3()

    model.eval()

    if optimize:
        # jit model to take it from ~20fps to ~30fps
        model = torch.jit.script(model)

        for param in model.parameters():
            param.grad = None

    
    
    if en_profiler:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):

                for i in range(niter):
                    for img in img_dir_list:
                        input_image = Image.open(img)

                        input_tensor = preprocess(input_image)
                        input_batch = input_tensor.unsqueeze(0) 

                        beg = perf_counter()
                        with torch.no_grad():
                            output = model(input_batch)
                        end = perf_counter()
                        diff = end - beg
                        print("Time: ", diff*1000)

                        results.append(return_n_biggest_result_pytorch(output_data=output, n_big=n_big, class_dir=label_dir))
                        inf_times.append(diff)

                prof.export_chrome_trace(os.path.join(json_path, model_dir))
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    else:
        for i in range(niter):
            for img in img_dir_list:
                input_image = Image.open(img)

                input_tensor = preprocess(input_image)
                input_batch = input_tensor.unsqueeze(0) 

                beg = perf_counter()
                with torch.no_grad():
                    output = model(input_batch)
                end = perf_counter()
                diff = end - beg
                print("Time in ms: ", diff*1000)

                results.append(return_n_biggest_result_pytorch(output_data=output, n_big=n_big, class_dir=label_dir))
                inf_times.append(diff)


    return results, inf_times
    
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

    if args.image:
        image_dir_list.append(args.image)
    elif args.image_folder:
        images = os.listdir(args.image_folder)

        for img in images:
            if ".jpg" in img and "._" not in img:
                image_dir_list.append(os.path.join(args.image_folder, img))
    else:
        quit("No img or image folder given")

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

    parser.add_argument("-op", "--output", help="where the results are saved", required=False, default="/home/pi/sambashare/BacArbeit/results/classification/")

    parser.add_argument("-s", '--sleep', default=1,type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-n", '--niter', default=1, type=int, help='number of iterations', required=False)
    parser.add_argument("--n_big", default=5)
    parser.add_argument("-url", "--pytorch_model_name", help="gives the name of the pytorch model which has to be downloaded from the internet", required=False)
    parser.add_argument("-opt", "--optimize", help="run optimzied inference code",required=False, action="store_true")
    parser.add_argument("-q", "--quantized", help="load quantized pytroch model", required=False, action="store_true")
    parser.add_argument("-bp", "--built_in_profiler", help="enable built in profiler", required=False, action="store_true")
    parser.add_argument("-cp", "--cprofiler", help="enable cProfiler", required=False, action="store_true")

    return parser.parse_args()

def build_dir_paths(args):
    general_dir = os.path.abspath(os.path.dirname(__file__)).split("scripts")[0]

    model_dir_list = handle_model_dir(args=args)
    img_dir_list = handle_img_dir(args=args)
    label_dir = handle_label_dir(args=args)
    inf_times_dir = os.path.join(args.output, "inference_time")
    result_dir = os.path.join(args.output, "prediction")
    c_profiler_dir = os.path.join(args.output, "cProfiler")


    return general_dir, model_dir_list, img_dir_list, label_dir, inf_times_dir, result_dir, c_profiler_dir

def handle_other_args_par(args):
    sleep = args.sleep
    niter = args.niter
    n_big = int(args.n_big)
    optimize = args.optimize
    built_in_profiler = args.built_in_profiler
    cprofiler = args.cprofiler
    quantized = args.quantized 

    return sleep, niter, n_big, optimize, built_in_profiler, cprofiler, quantized
    
def main():
    
    profiler = cProfile.Profile()

    args = handle_arguments()
    general_dir, model_dir_list, img_dir_list, label_dir, inf_times_dir, result_dir, c_profiler_dir = build_dir_paths(args=args)
    sleep, niter, n_big, optimize, built_in_profiler, cprofiler, quantized = handle_other_args_par(args=args)
    
    print("\n")
    print("Model list: ", model_dir_list)
    print("Image list: ", img_dir_list)
    print("\n")

    if args.api == "tflite_runtime":
        check_directories(model_dir_list, img_dir_list, ".tflite")

        for model in model_dir_list:
            model_name = model.split("/")[-1].split(".tflite")[0] + "_tflite_runtime.txt"
            inf_times_file = os.path.join(inf_times_dir, model_name)
            result_file = os.path.join(result_dir, model_name)

            if cprofiler:
                c_profiler_file = os.path.join(c_profiler_dir, model_name)
                profiler.enable()

            results, inf_times = tflite_runtime(model, img_dir_list, label_dir, n_big, niter, optimize=optimize)
            
            if cprofiler:
                profiler.disable()

                with open(c_profiler_file, 'w') as stream:
                    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                    stats.print_stats()

            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")
            
            with open(inf_times_file, "w") as file:
                for r in inf_times:
                    file.writelines(str(r))
                    file.writelines("\n")
     
    elif args.api == "pyarmnn":
        check_directories(model_dir_list, img_dir_list, ".tflite")

        for model in model_dir_list:
            csv_path = os.path.join(args.output, "pyarmnn_profiler")
            model_name_txt = model.split("/")[-1].split(".tflite")[0] + "_pyarmnn.txt"
            model_name_csv = model.split("/")[-1].split(".tflite")[0] + "_pyarmnn.csv"
            inf_times_file = os.path.join(inf_times_dir, model_name_txt)
            result_file = os.path.join(result_dir, model_name_txt)
            csv_path = os.path.join(csv_path, model_name_csv)
            
            if cprofiler:
                c_profiler_file = os.path.join(c_profiler_dir, model_name_txt)
                profiler.enable()

            results, inf_times = pyarmnn(model, img_dir_list, label_dir, n_big, niter, csv_path, built_in_profiler)

            if cprofiler:
                profiler.disable()

                with open(c_profiler_file, 'w') as stream:
                    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                    stats.print_stats()

            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")
            
            with open(inf_times_file, "w") as file:
                for r in inf_times:
                    file.writelines(str(r))
                    file.writelines("\n")

    elif args.api == "onnx":
        check_directories(model_dir_list, img_dir_list, ".onnx")

        for model in model_dir_list:
            json_path = os.path.join(args.output, "onnx_profiler")
            model_name_txt = model.split("/")[-1].split(".onnx")[0] + "_onnx.txt"
            inf_times_file = os.path.join(inf_times_dir, model_name_txt)
            result_file = os.path.join(result_dir, model_name_txt)

            if cprofiler:
                c_profiler_file = os.path.join(c_profiler_dir, model_name_txt)
                profiler.enable()

            results, inf_times = onnx_runtime(model, img_dir_list, label_dir, n_big, niter, json_path, optimize, built_in_profiler)

            if cprofiler:
                profiler.disable()
                with open(c_profiler_file, 'w') as stream:
                    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                    stats.print_stats()

            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")
            
            with open(inf_times_file, "w") as file:
                for r in inf_times:
                    file.writelines(str(r))
                    file.writelines("\n")

    elif args.api == "pytorch":
        for model in model_dir_list:
            json_path = os.path.join(args.output, "pytorch_profiler")
            if args.pytorch_model_name:
                model_name_txt = args.pytorch_model_name + "_pytorch.txt"
            else:
                model_name_txt = model.split("/")[-1].split(".pth")[0] + "_pytorch.txt"
            inf_times_file = os.path.join(inf_times_dir, model_name_txt)
            result_file = os.path.join(result_dir, model_name_txt) 

            if cprofiler:
                c_profiler_file = os.path.join(c_profiler_dir, model_name_txt)
                profiler.enable()

            results, inf_times = pytorch(model, img_dir_list, label_dir, n_big, niter, json_path, optimize, built_in_profiler, quantized)

            if cprofiler:
                profiler.disable()
                with open(c_profiler_file, 'w') as stream:
                    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                    stats.print_stats()

            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")
            
            with open(inf_times_file, "w") as file:
                for r in inf_times:
                    file.writelines(str(r))
                    file.writelines("\n")
    
    elif args.api == "ov":
        #check_directories(model_dir = model_dir_list, img_dir= img_dir_list, model_type=".xml")
    
        for model in model_dir_list:
            model_name_txt = model.split("/")[-1].split(".xml")[0] + "_ov.txt"
            inf_times_file = os.path.join(inf_times_dir, model_name_txt)
            result_file = os.path.join(result_dir, model_name_txt) 

            if cprofiler:
                c_profiler_file = os.path.join(c_profiler_dir, model_name_txt)
                profiler.enable()


            #results, inf_times = openvino(model, img_dir_list, label_dir, n_big, niter, optimize)

            if optimize:

                with open("async_ov.txt", "w") as file:
                    file.writelines("")

                results, inf_times = async_openvino(model, img_dir_list, label_dir, n_big, niter)

                os.replace("async_ov.txt", result_file)
            else:
                results, inf_times = sync_openvino(model, img_dir_list, label_dir, n_big, niter)

                with open(result_file, "w") as file:
                    for r in results:
                        file.writelines(str(r))
                        file.writelines("\n")

            if cprofiler:
                profiler.disable()
                with open(c_profiler_file, 'w') as stream:
                    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                    stats.print_stats()
            
            with open(inf_times_file, "w") as file:
                for r in inf_times:
                    file.writelines(str(r))
                    file.writelines("\n")


if __name__ == "__main__":
    main()

