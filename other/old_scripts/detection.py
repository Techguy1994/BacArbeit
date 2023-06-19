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

def get_result(class_dir, probabilities):
    with open(class_dir, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

def return_n_biggest_result_pytorch(output_data, n_big=10):
    max_positions = np.argpartition(output_data, -n_big)[-n_big:]
    out_normalization_factor = 1

    #print(output_details[0]["dtype"])

    #if "integer" in output_details:
    #    print("int")
    #    quit("no adapted to onnx, please change following code when quantized model is given")
    #    out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    #elif "float" in output_details:
    #    print("float")
    #    out_normalization_factor = 1

    result = {}

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[entry] / out_normalization_factor
        result[entry] = [val*100]
        print("\tpos {} : {:.2f}%".format(entry, val*100))
        
    return result

def return_n_biggest_result_ov(output_data, n_big=10):
    max_positions = np.argpartition(output_data, -n_big)[-n_big:]
    out_normalization_factor = 1

    #print(output_details[0]["dtype"])

    #if "integer" in output_details:
    #    print("int")
    #    quit("no adapted to onnx, please change following code when quantized model is given")
    #    out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    #elif "float" in output_details:
    #    print("float")
    #    out_normalization_factor = 1

    result = {}

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[entry] / out_normalization_factor
        result[entry] = [val*100]
        print("\tpos {} : {:.2f}%".format(entry, val*100))
        
    return result

def return_n_biggest_result_pyarmnn(output_data, n_big=3):

    max_positions = np.argpartition(output_data[0], -n_big)[-n_big:]

    if output_data.dtype == "uint8":
        out_normalization_factor = np.iinfo(output_data.dtype).max
    elif output_data.dtype == "float32":
        out_normalization_factor = 1
    
    result = {}

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[0][entry] / out_normalization_factor
        result[entry] = [val*100]
        print("\tpos {} : {:.2f}%".format(entry, val*100))
        
    return result

def output_data_tflite_runtime(output_details, interpreter, img_org, thres, img_result_file, class_dir):
        
        results = []

        with open(class_dir, "r") as f:
            categories = [s.strip() for s in f.readlines()]

        # get outpu tensor
        boxes = interpreter.get_tensor(output_details[0]['index'])
        labels = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num = interpreter.get_tensor(output_details[3]['index'])
        
        for i in range(boxes.shape[1]):
            if scores[0, i] > thres:
                box = boxes[0, i, :]
                x0 = int(box[1] * img_org.shape[1])
                y0 = int(box[0] * img_org.shape[0])
                x1 = int(box[3] * img_org.shape[1])
                y1 = int(box[2] * img_org.shape[0])
                box = box.astype(int)
                cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
                cv2.putText(img_org,
                    str(int(labels[0, i])) + ": " + categories[int(labels[0,i])-1],
                    (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)
                results.append({"object": categories[int(labels[0,i])-1],"index": int(labels[0,i]), "accuracy": scores[0,i]})
                
        
        cv2.imwrite(img_result_file, img_org)
        return results

def output_data_pyarmnn(output_details, img_org, thres, img_result_file, class_dir):
        
        results = []

        with open(class_dir, "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # get outpu tensor
        boxes = output_details[0]
        labels = output_details[1]
        scores = output_details[2]
        num = output_details[3]

        print(boxes.shape, labels.shape, scores.shape, num.shape)

        
        for i in range(boxes.shape[1]):
            if scores[0, i] > thres:
                box = boxes[0, i, :]
                x0 = int(box[1] * img_org.shape[1])
                y0 = int(box[0] * img_org.shape[0])
                x1 = int(box[3] * img_org.shape[1])
                y1 = int(box[2] * img_org.shape[0])
                box = box.astype(int)
                cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
                cv2.putText(img_org,
                    str(int(labels[0, i])) + ": " + categories[int(labels[0,i])-1],
                    (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)
                results.append({"object": categories[int(labels[0,i])-1],"index": int(labels[0,i]), "accuracy": scores[0,i]})
                
        
        cv2.imwrite(img_result_file, img_org)
        return results

def output_data_pytorch_yolov5(output_details, img_org, thres, img_result_file, class_dir):
    #https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd-mobilenetv1
        results = []

        with open(class_dir, "r") as f:
            categories = [s.strip() for s in f.readlines()]
        
        #print(output_details)

        #boxes, labels, scores, num = output_details

        boxes = output_details[:,0:4].numpy()
        scores = output_details[:,4].numpy()
        labels = output_details[:,5].numpy()

        print(boxes.shape, labels.shape, scores.shape)
        #print("num", num)
        print("classes", labels)
        print("Scores", scores)

        
        print("Start")
        print(boxes)
        #boxes = boxes.numpy()
        print(boxes)
        print("End")

        #print("Number: ", num.shape)

        # get outpu tensor
        #boxes = output_details[0]
        #labels = output_details[1]
        #scores = output_details[2]
        #num = output_details[3]

        print("1:", boxes.shape[1])
        print("0:", boxes.shape[0])
        
        for i in range(boxes.shape[0]):
            if scores[i] > thres:
                box = boxes[i, :]
                print(box[0], box[1], box[2], box[3])
                #x0 = int(box[1] * img_org.shape[1])
                #y0 = int(box[0] * img_org.shape[0])
                #x1 = int(box[3] * img_org.shape[1])
                #y1 = int(box[2] * img_org.shape[0])

                x0 = int(box[0])
                y0 = int(box[1]) 
                x1 = int(box[2])
                y1 = int(box[3]) 
                
                
                #x, y, w, h = box[1], box[0], box[3], box[2]
                #x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
                #xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

                print(x0, y0, x1, y1)
                box = box.astype(int)
                cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
                cv2.putText(img_org,
                    str(int(labels[i])) + ": " + categories[int(labels[i])],
                    (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)
                results.append({"object": categories[int(labels[i])],"index": int(labels[i]), "accuracy": scores[i]})
                
        
        cv2.imwrite(img_result_file, img_org)
        return results

def output_data_onnx(output_details, img_org, thres, img_result_file, class_dir):
    #https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd-mobilenetv1
        results = []

        with open(class_dir, "r") as f:
            categories = [s.strip() for s in f.readlines()]
        
        #print(output_details)

        boxes, labels, scores, num = output_details
        print(boxes.shape, labels.shape, scores.shape, num.shape)
        print("num", num)
        print("classes", labels)
        print("Scores", scores)

        #print("Number: ", num.shape)

        # get outpu tensor
        #boxes = output_details[0]
        #labels = output_details[1]
        #scores = output_details[2]
        #num = output_details[3]

        #print(boxes.type)
        
        for i in range(boxes.shape[1]):
            if scores[0, i] > thres:
                box = boxes[0, i, :]
                x0 = int(box[1] * img_org.shape[1])
                y0 = int(box[0] * img_org.shape[0])
                x1 = int(box[3] * img_org.shape[1])
                y1 = int(box[2] * img_org.shape[0])
                box = box.astype(int)
                cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
                cv2.putText(img_org,
                    str(int(labels[0, i])) + ": " + categories[int(labels[0,i])],
                    (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)
                results.append({"object": categories[int(labels[0,i])],"index": int(labels[0,i]), "accuracy": scores[0,i]})
                
        
        cv2.imwrite(img_result_file, img_org)
        return results

def output_data_ov(output_details, img_org, thres, img_result_file, class_dir):
    #https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd-mobilenetv1
        results = []

        with open(class_dir, "r") as f:
            categories = [s.strip() for s in f.readlines()]
        
        for element in output_details:
            print("\n")
            print(element, ":-> \n", output_details[element])
            output = output_details[element][0][0]

        print(output.shape)

        #image_id = output[:,0]
        labels = output[:,1]
        scores = output[:,2]
        boxes = output[:,3:7]

        #print(boxes.type)
        print("head labels", labels.shape, "content: ", labels)
        
        for i in range(boxes.shape[0]):

            if scores[i] > thres:
                #print("labels:", labels[i])
                label = labels[i]
                box = boxes[i, :]
                x0 = int(box[0] * img_org.shape[1])
                y0 = int(box[1] * img_org.shape[0])
                x1 = int(box[2] * img_org.shape[1])
                y1 = int(box[3] * img_org.shape[0])
                box = box.astype(int)
                cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
                cv2.putText(img_org,
                    str(int(label)) + ": " + categories[int(label)],
                    (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)
                results.append({"object": categories[int(labels[i])],"index": int(labels[i]), "accuracy": scores[i]})
                
        
        cv2.imwrite(img_result_file, img_org)
        return results

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess_image_pytorch_pytorch_v5s():

    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    return preprocess

def preprocess_image_ov_ssd_mobilenet_v1(input_tensor, model):
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

def preprocess_image_onnx_ssd_mobilenet_v1(image_path, input_type, channels=3):
    img = Image.open(image_path)

    img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], channels)
    img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
    return img_data


def preprocess_image_tflite_ssd_mobilenet_v1(image_path, height, width, input_type, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS)
    image_data = np.asarray(image).astype(input_type)
    #for channel in range(image_data.shape[0]):
    #    image_data[channel, :, :] = image_data[channel, :, :]*2 / 255 - 1
    image_data = np.expand_dims(image_data, 0)
    #print(image_data.shape)
    #quit()
    return image_data



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

def tflite_runtime(model_dir, img_dir, label_dir, thres, niter, img_result_dir):
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
            img_org = cv2.imread(img)
            img = preprocess_image_tflite_ssd_mobilenet_v1(img, input_shape[1], input_shape[2], input_type)

            interpreter.set_tensor(input_details[0]['index'], img)

            beg = time()
            interpreter.invoke()
            end = time()
            inf_time = end-beg
            inf_times.append(inf_time*1000)
            print(inf_time*1000)

            results.append(output_data_tflite_runtime(output_details, interpreter, img_org, thres, img_result_file, label_dir))
    
    return results
    

def pyarmnn(model_dir, img_dir, label_dir, thres, niter, csv_path, img_result_dir):

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

    input_type = float

    if ann.TensorInfo.IsQuantized(input_tensor_info):
        print(np.uint8)
        input_type = np.uint8
    
    #inference 
    results = []
    for i in range(niter):
        for img in img_dir:
            img_result_file = os.path.join(img_result_dir, img.split("/")[-1])
            img_org = cv2.imread(img)

            image = preprocess_image_tflite_ssd_mobilenet_v1(img, height, width, input_type)

            input_tensors = ann.make_input_tensors([input_binding_info], [image])
            runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
            result = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
            results.append(output_data_pyarmnn(result, img_org, thres, img_result_file, label_dir))

    write_profiling_data_pyarmnn(profiler, model_dir, csv_path)

    return results

def openvino(model_dir, img_dir, label_dir, thres, niter, img_result_dir):

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
            # Add N dimension
            input_tensor = np.expand_dims(image, 0)
            print(input_tensor.shape)

            # Preprpocess
            input_tensor, model = preprocess_image_ov_ssd_mobilenet_v1(input_tensor, model)


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
            results.append(output_data_ov(result, img_org, thres, img_result_file, label_dir))

    return results

def onnx_runtime(model_dir, img_dir_list, label_dir, thres, niter, json_path, img_result_dir):
                 
    results = []

    options = onnxruntime.SessionOptions()
    options.enable_profiling = True

    session = onnxruntime.InferenceSession(model_dir, options)

    input_name = session.get_inputs()[0].name
    print(session.get_outputs()[0].name, session.get_outputs()[1].name, session.get_outputs()[2].name, session.get_outputs()[3].name)
    outputs = [session.get_outputs()[0].name, session.get_outputs()[1].name, session.get_outputs()[2].name, session.get_outputs()[3].name]
    print(outputs)
    #output_name = session.get_outputs()[0].name
    #print(output_name)

    #outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]

    image_height = session.get_inputs()[0].shape[2]
    image_width = session.get_inputs()[0].shape[3]

    input_data_type = session.get_inputs()[0].type
    output_data_type = session.get_outputs()[0].type

    for i in range(niter):
        for img in img_dir_list:
            img_result_file = os.path.join(img_result_dir, img.split("/")[-1])
            img_org = cv2.imread(img)
            output = session.run(outputs, {input_name:preprocess_image_onnx_ssd_mobilenet_v1(img, input_data_type)})
            #print("Output: ", output.shape)
            #print("Output 0: ", output[0].shape)




            #output = session.run([output_name], {input_name: img_data})[0]
            #quit()

            #output = output.flatten()
            #output = softmax(output) # this is optional
            results.append(output_data_onnx(output, img_org, thres, img_result_file, label_dir))
        
    prof_file = session.end_profiling()
    print(prof_file)

    os.rename(prof_file, os.path.join(json_path, prof_file))
      
    return results


def pytorch(model_dir, img_dir_list, label_dir, thres, niter, json_path, img_result_dir):
    print("Pytorch")

    results = []

    model = torch.hub.load('ultralytics/yolov5', model_dir, pretrained=True)
    #model.eval()

    preprocess = preprocess_image_pytorch_pytorch_v5s()

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):

            for i in range(niter):
                for img in img_dir_list:
                    img_result_file = os.path.join(img_result_dir, img.split("/")[-1])
                    #input_image = Image.open(img)

                    #img = 'https://ultralytics.com/images/zidane.jpg'
                    img_org = img_org = cv2.imread(img)

                    #input_tensor = preprocess(input_image)
                    #input_batch = input_tensor.unsqueeze(0) 

                    start_time = time()
                    with torch.no_grad():
                        output = model(img)
                    end_time = time()
                    print(end_time-start_time)
                    output = output.xyxy[0]
                    print(output)
                    #output.print()

                    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
                    #probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    #get_result(label_dir, probabilities)

                    results.append(output_data_pytorch_yolov5(output, img_org, thres, img_result_file, label_dir))

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

    parser.add_argument("-o", "--output", help="where the results are saved", required=False, default="/home/pi/sambashare/BacArbeit/results/detection/")

    parser.add_argument("-s", '--sleep', default=1,type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-n", '--niter', default=1, type=int, help='number of iterations', required=False)
    parser.add_argument("-thres", "--threshold", default=0.5)
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


    return general_dir, model_dir_list, img_dir_list, label_dir, inf_times_dir, result_dir, img_result_dir

def handle_other_args_par(args):
    sleep = args.sleep
    niter = args.niter
    thres = float(args.threshold)

    return sleep, niter, thres
    

def main():
    
    profiler = cProfile.Profile()

    args = handle_arguments()
    general_dir, model_dir_list, img_dir_list, label_dir, inf_times_dir, result_dir, img_result_dir = build_dir_paths(args=args)
    sleep, niter, thres = handle_other_args_par(args=args)

    print(model_dir_list)

    if args.api == "tflite_runtime":
        check_directories(model_dir_list, img_dir_list, ".tflite")
        
        for model in model_dir_list:
            model_name = model.split("/")[-1].split(".tflite")[0] + "_tflite_runtime.txt"
            inf_times_file = os.path.join(inf_times_dir, model_name)
            result_file = os.path.join(result_dir, model_name)
            profiler.enable()
            results = tflite_runtime(model, img_dir_list, label_dir, thres, niter, img_result_dir)
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
            results = pyarmnn(model, img_dir_list, label_dir, thres, niter, csv_path, img_result_dir)
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
            results = onnx_runtime(model, img_dir_list, label_dir, thres, niter, json_path, img_result_dir)
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
            results = pytorch(model, img_dir_list, label_dir, thres, niter, json_path, img_result_dir)
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
            results = openvino(model, img_dir_list, label_dir, thres, niter, img_result_dir)
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

