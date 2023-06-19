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

def return_n_biggest_result_tflite_runtime(output_data, output_details, n_big=10):
    max_positions = np.argpartition(output_data[0], -n_big)[-n_big:]
    out_normalization_factor = 1

    print(output_details[0]["dtype"])

    if output_details[0]['dtype'] == np.uint8:
        print("int")
        out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    elif output_details[0]['dtype'] == np.float32:
        print("float")
        out_normalization_factor = 1

    result = {}

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[0][entry] / out_normalization_factor
        result[entry] = [val*100]
        print("\tpos {} : {:.2f}%".format(entry, val*100))
        
    return result

def return_n_biggest_result_onnx(output_data, output_details, n_big=10):
    max_positions = np.argpartition(output_data, -n_big)[-n_big:]
    out_normalization_factor = 1

    #print(output_details[0]["dtype"])

    if "integer" in output_details:
        print("int")
        quit("no adapted to onnx, please change following code when quantized model is given")
        out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    elif "float" in output_details:
        print("float")
        out_normalization_factor = 1

    result = {}

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[entry] / out_normalization_factor
        result[entry] = [val*100]
        print("\tpos {} : {:.2f}%".format(entry, val*100))
        
    return result

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess_image_pytorch_mobilenet2():

    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    return preprocess

def preprocess_image_onnx_mobilenet2(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS)
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = image_data[channel, :, :] / 255 - mean[channel] / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

def preprocess_image_tflite_mobilenet2(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS)
    image_data = np.asarray(image).astype(np.float32)
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = image_data[channel, :, :]*2 / 255 - 1
    image_data = np.expand_dims(image_data, 0)
    return image_data

def load_image(height, width, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (height, width))
    cv2.imwrite("resized_input.jpg", img)
    img = np.expand_dims(img, axis=0)
    return img


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


    """
    if len(model_dir) == 0:
        quit("empty model dir")
    elif len(img_dir) == 0:
        quit("empty img dir")

    if model_type not in model_dir:
        quit("wrong model given")
    """
    

    
def return_picture_list(img_dir):
    print(img_dir)
    img_list = []
    pictures = os.listdir(img_dir)
    print(pictures)

    for picture in pictures:
        if any(end in picture for end in [".jpg", ".png"]):
            img_list.append(os.path.join(img_dir, picture))
    
    return img_list


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

def tflite_runtime(model_dir, img_dir, label_dir, n_big, niter):
    #source: 
    #https://www.tensorflow.org/lite/guide/inference
    #https://github.com/NXPmicro/pyarmnn-release/tree/master/python/pyarmnn/examples
    print("tflite")

    results = []
    inf_times = []


    #model_dir = os.path.join(model_dir, "tflite")
    check_directories(model_dir, img_dir, ".tflite")
    #model_dir = os.path.join(model_dir, os.listdir(model_dir)[0])
    #img_list = return_picture_list(img_dir)
    
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

            img = preprocess_image_tflite_mobilenet2(img, input_shape[1], input_shape[2])

            #if input_type == np.uint8:
            #    print(np.iinfo(input_type).max)
            #    img = np.uint8(img)
            #else:
            #   img = np.float32(img/np.iinfo("uint8").max)

            #input_data = np.array(img, dtype=input_type)
            interpreter.set_tensor(input_details[0]['index'], img)

            beg = time()
            interpreter.invoke()
            end = time()
            inf_time = end-beg
            inf_times.append(inf_time*1000)
            print(inf_time*1000)

            output_data = interpreter.get_tensor(output_details[0]['index'])

            results.append(return_n_biggest_result_tflite_runtime(output_data=output_data, output_details=output_details, n_big=n_big))
            print(results)
    
    return results


    

def pyarmnn(model_dir, img_dir, label_dir, n_big, niter, csv_path):

    print("pyarmnn")
    # LINK TO CODE: https://www.youtube.com/watch?v=HQYosuy4ABY&t=1867s
    #https://developer.arm.com/documentation/102557/latest
    #file:///C:/Users/Maroun_Desktop_PC/SynologyDrive/Bachelorarbeit/pyarmnn/pyarmnn_doc.html#pyarmnn.IOutputSlot

    results = []

    #model_dir = os.path.join(model_dir, "tflite")
    check_directories(model_dir, img_dir, ".tflite")
    #model_dir = os.path.join(model_dir, os.listdir(model_dir)[0])

    #img_list = return_picture_list(img_dir)

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
    
    #inference 
    results = []
    for i in range(niter):
        for img in img_dir:
            #image = load_image(width, height, img)

            #if ann.TensorInfo.IsQuantized(input_tensor_info):
            #    image = np.uint8(image)
            #else:
            #    image = np.float32(image/np.iinfo("uint8").max)

            image = preprocess_image_tflite_mobilenet2(img, height, width)

            input_tensors = ann.make_input_tensors([input_binding_info], [image])
            runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
            result = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
            result = return_n_biggest_result_pyarmnn(result[0], n_big)
            results.append(result)

    write_profiling_data_pyarmnn(profiler, model_dir, csv_path)

    return results

def openvino(model_dir, img_dir, label_dir, n_big, niter):

    results = []
    print("openvino")
    print(model_dir)

    device_name = "CPU"

    #check_directories(model_dir, img_dir, ".xml")



    #for entry in os.listdir(model_dir):
    #    if ".xml" in entry:
    #        model_dir = os.path.join(model_dir, entry)

    #print(model_dir)

    #img_list = return_picture_list(img_dir)

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

# --------------------------- Step 5. Loading model to the device -----------------------------------------------------
        log.info('Loading the model to the plugin')
        compiled_model = core.compile_model(model, device_name)

# --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
        log.info('Starting inference in synchronous mode')
        result = compiled_model.infer_new_request({0: input_tensor})

# --------------------------- Step 7. Process output ------------------------------------------------------------------
        predictions = next(iter(result.values()))
        probs = predictions.reshape(-1)
        print(probs.shape)
        #print(predictions[0][:])
        results.append(return_n_biggest_result_ov(probs, n_big))

        # Change a shape of a numpy.ndarray with results to get another one with one dimension
        

        # Get an array of 10 class IDs in descending order of probability
        top_10 = np.argsort(probs)[-10:][::-1]

        header = 'class_id probability'

        log.info(f'Image path: {img}')
        log.info('Top 10 results: ')
        log.info(header)
        log.info('-' * len(header))

        for class_id in top_10:
            probability_indent = ' ' * (len('class_id') - len(str(class_id)) + 1)
            log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}')
            print(f'{class_id}{probability_indent}{probs[class_id]:.7f}')

        log.info('')

    return results

def onnx_runtime(model_dir, img_dir_list, label_dir, n_big, niter, json_path):
                 
    results = []
    #model_dir = os.path.join(model_dir, "onnx")
    #print(model_dir)

    check_directories(model_dir, img_dir_list, ".onnx")
    #model_dir = os.path.join(model_dir, os.listdir(model_dir)[0])
    #img_list = return_picture_list(img_dir)

    options = onnxruntime.SessionOptions()
    options.enable_profiling = True

    session = onnxruntime.InferenceSession(model_dir, options)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image_height = session.get_inputs()[0].shape[2]
    image_width = session.get_inputs()[0].shape[3]

    input_data_type = session.get_inputs()[0].type
    output_data_type = session.get_outputs()[0].type

    #print(input_data_type, output_data_type)

    for i in range(niter):
        for img in img_dir_list:
            output = session.run([output_name], {input_name:preprocess_image_onnx_mobilenet2(img, image_height, image_width)})[0]
            output = output.flatten()
            output = softmax(output) # this is optional
            results.append(return_n_biggest_result_onnx(output, output_data_type, n_big))
        
    prof_file = session.end_profiling()
    print(prof_file)

    os.rename(prof_file, os.path.join(json_path, prof_file))
      
    return results



def pytorch(model_dir, img_dir_list, label_dir, n_big, niter, json_path):
    print("Pytorch")

    results = []

    print(model_dir)

    #check_directories(model_dir, img_dir_list, ".pth")

    model = torch.hub.load('pytorch/vision:v0.10.0', model_dir, pretrained=True)
    model.eval()

    preprocess = preprocess_image_pytorch_mobilenet2()

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):

            for i in range(niter):
                for img in img_dir_list:
                    input_image = Image.open(img)

                    input_tensor = preprocess(input_image)
                    input_batch = input_tensor.unsqueeze(0) 

                    start_time = time()
                    with torch.no_grad():
                        output = model(input_batch)
                    end_time = time()
                    print(end_time-start_time)

                    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
                    #print(output[0])
                    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    #print(probabilities)
                    get_result(label_dir, probabilities)

                    results.append(return_n_biggest_result_pytorch(output_data=probabilities, n_big=n_big))


                    # Read the categories

    prof.export_chrome_trace(os.path.join(json_path, model_dir))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    return results
    

def handle_model_dir(args):

    model_dir_list = []

    if args.pytorch_model_name and args.api == "pytorch":
        model_dir_list.append(args.pytorch_model_name)
    else:
        print("hey")
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

    parser.add_argument("-o", "--output", help="where the results are saved", required=False, default="/home/pi/sambashare/BacArbeit/results/classification/")

    parser.add_argument("-s", '--sleep', default=1,type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-n", '--niter', default=1, type=int, help='number of iterations', required=False)
    parser.add_argument("--n_big", default=5)
    parser.add_argument("-url", "--pytorch_model_name", help="gives the name of the pytorch model which has to be downloaded from the internet", required=False)

    return parser.parse_args()

def build_dir_paths(args):
    general_dir = os.path.abspath(os.path.dirname(__file__)).split("scripts")[0]

    model_dir_list = handle_model_dir(args=args)
    img_dir_list = handle_img_dir(args=args)
    label_dir = handle_label_dir(args=args)
    inf_times_dir = os.path.join(args.output, "inference_time")
    result_dir = os.path.join(args.output, "prediction")


    return general_dir, model_dir_list, img_dir_list, label_dir, inf_times_dir, result_dir

def handle_other_args_par(args):
    sleep = args.sleep
    niter = args.niter
    n_big = args.n_big

    return sleep, niter, n_big
    

def main():
    
    profiler = cProfile.Profile()

    args = handle_arguments()
    general_dir, model_dir_list, img_dir_list, label_dir, inf_times_dir, result_dir = build_dir_paths(args=args)
    sleep, niter, n_big = handle_other_args_par(args=args)

    print(model_dir_list)

    if args.api == "tflite_runtime":
        for model in model_dir_list:
            model_name = model.split("/")[-1].split(".tflite")[0] + "_tflite_runtime.txt"
            inf_times_file = os.path.join(inf_times_dir, model_name)
            result_file = os.path.join(result_dir, model_name)

            profiler.enable()
            results = tflite_runtime(model, img_dir_list, label_dir, n_big, niter)
            profiler.disable()

            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")

            with open(inf_times_file, 'w') as stream:
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats()

            
    elif args.api == "pyarmnn":
        for model in model_dir_list:
            csv_path = os.path.join(args.output, "pyarmnn_profiler")
            model_name_txt = model.split("/")[-1].split(".tflite")[0] + "_pyarmnn.txt"
            model_name_csv = model.split("/")[-1].split(".tflite")[0] + "_pyarmnn.csv"
            inf_times_file = os.path.join(inf_times_dir, model_name_txt)
            result_file = os.path.join(result_dir, model_name_txt)
            csv_path = os.path.join(csv_path, model_name_csv)

            profiler.enable()
            results = pyarmnn(model, img_dir_list, label_dir, n_big, niter, csv_path)
            profiler.disable()

            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")

            with open(inf_times_file, 'w') as stream:
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats()

    elif args.api == "onnx":
        for model in model_dir_list:
            json_path = os.path.join(args.output, "onnx_profiler")
            model_name_txt = model.split("/")[-1].split(".onnx")[0] + "_onnx.txt"
            inf_times_file = os.path.join(inf_times_dir, model_name_txt)
            result_file = os.path.join(result_dir, model_name_txt)

            profiler.enable()
            results = onnx_runtime(model, img_dir_list, label_dir, n_big, niter, json_path)
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
            results = pytorch(model, img_dir_list, label_dir, n_big, niter, json_path)
            profiler.disable()

            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")

            with open(inf_times_file, 'w') as stream:
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats()
    
    elif args.api == "ov":
        print(model_dir_list)
        check_directories(model_dir = model_dir_list, img_dir= img_dir_list, model_type=".xml")
        print(model_dir_list)
        for model in model_dir_list:
            model_name_txt = model.split("/")[-1].split(".xml")[0] + "_ov.txt"
            inf_times_file = os.path.join(inf_times_dir, model_name_txt)
            result_file = os.path.join(result_dir, model_name_txt) 

            profiler.enable()
            results = openvino(model, img_dir_list, label_dir, n_big, niter)
            profiler.disable()

            with open(result_file, "w") as file:
                for r in results:
                    file.writelines(str(r))
                    file.writelines("\n")

            with open(inf_times_file, 'w') as stream:
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats()





  


"""
    parser.add_argument("-tfl", '--tflite_model', help='TFLite model file', required=False)
    parser.add_argument("-sd", '--save_dir', default='./tmp', help='folder to save the resulting files', required=False)
    parser.add_argument("-n", '--niter', default=10, type=int, help='number of iterations', required=False)
    parser.add_argument("-s", '--sleep', default=0, type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-thr", '--threads', default=1, type=int, help='number of threads to run compiled bench on', required=False)
    parser.add_argument("-bf", '--bench_file', default="linux_aarch64_benchmark_model", type=str, help='path to compiled benchmark file', required=False)

    parser.add_argument('--print', dest='print', action='store_true')
    parser.add_argument('--no-print', dest='print', action='store_false')
    parser.set_defaults(feature=False)

    parser.add_argument('--interpreter', dest='interpreter', action='store_true')
    parser.add_argument('--no-interpreter', dest='interpreter', action='store_false')
    parser.set_defaults(feature=False)

    parser.add_argument('--pyarmnn', dest='pyarmnn', action='store_true')
    parser.add_argument('--no-pyarmnn', dest='pyarmnn', action='store_false')
    parser.set_defaults(feature=False)

    args = parser.parse_args()
    
    if not args.pyarmnn and not args.interpreter and not args.bench_file:
        logging.error("No Runtime chosen, please choose either PyARMNN, the TFLite Interpreter or provide a compiled benchmark file")
        return

    # if TFLite model is provided, use it for inference
    if args.tflite_model and os.path.isfile(args.tflite_model):
        #run_network(tflite_path=args.tflite_model, save_dir=args.save_dir, niter=args.niter,
        #                print_bool=args.print, sleep_time=args.sleep, use_tflite=args.interpreter, use_pyarmnn=args.pyarmnn)
        print("tflite")
    else:
        # if no neural network models are provided, return
        logging.error("Invalid model path {} passed.".format(args.tflite_model))
        return

    # run inference using the provided benchmark file if the benchmark file is valid
    if args.bench_file and os.path.isfile(args.bench_file):
        #run_compiled_bench(tflite_path=args.tflite_model, save_dir=args.save_dir, niter=args.niter, print_bool=args.print, 
        #sleep_time=args.sleep, use_tflite=args.interpreter, use_pyarmnn=args.pyarmnn, bench_file=args.bench_file, num_threads = args.threads)
        print("benchmark")
    
      #logging.info("\n**********RPI INFERENCE DONE**********")

    
"""


if __name__ == "__main__":
    main()

