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


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess_image(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS)
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

def setup_profiling(net_id, runtime):
    profiler = runtime.GetProfiler(net_id)
    profiler.EnableProfiling(True)
    return profiler

def check_directories(model_dir, img_dir):
    if len(os.listdir(model_dir)) == 0:
        print("empty model dir")
    elif len(os.listdir(img_dir)) == 0:
        print("empty img dir")
    else:
        print("not empty")
    
def return_picture_list(img_dir):
    print(img_dir)
    img_list = []
    pictures = os.listdir(img_dir)
    print(pictures)

    for picture in pictures:
        if any(end in picture for end in [".jpg", ".png"]):
            img_list.append(os.path.join(img_dir, picture))
    
    return img_list

def load_image(height, width, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (height, width))
    cv2.imwrite("resized_input.jpg", img)
    img = np.expand_dims(img, axis=0)
    return img

def get_result(class_dir, probabilities):
    with open(class_dir, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())



def tflite_runtime(model_dir, img_dir):
    #source: 
    #https://www.tensorflow.org/lite/guide/inference
    #https://github.com/NXPmicro/pyarmnn-release/tree/master/python/pyarmnn/examples
    print("tflite")

    results = []
    inf_times = []
    img_list = []

    model_dir = os.path.join(model_dir, "tflite")
    check_directories(model_dir, img_dir)
    model_dir = os.path.join(model_dir, os.listdir(model_dir)[0])

    img_list = return_picture_list(img_dir)
    
    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_dir)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']

    for img in img_list:
        print(img)

        img = load_image(input_shape[1], input_shape[2], img)

        if input_type == np.uint8:
            print(np.iinfo(input_type).max)
            img = np.uint8(img)
        else:
            img = np.float32(img/np.iinfo("uint8").max)

        input_data = np.array(img, dtype=input_type)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        beg = time()
        interpreter.invoke()
        end = time()
        inf_time = end-beg
        inf_times.append(inf_time*1000)
        print(inf_time*1000)

        output_data = interpreter.get_tensor(output_details[0]['index'])


    

def pyarmnn(model_dir, img_dir):
    # LINK TO CODE: https://www.youtube.com/watch?v=HQYosuy4ABY&t=1867s
    #https://developer.arm.com/documentation/102557/latest
    #file:///C:/Users/Maroun_Desktop_PC/SynologyDrive/Bachelorarbeit/pyarmnn/pyarmnn_doc.html#pyarmnn.IOutputSlot

    results = []
    inf_times = []
    img_list = []

    model_dir = os.path.join(model_dir, "tflite")
    check_directories(model_dir, img_dir)
    model_dir = os.path.join(model_dir, os.listdir(model_dir)[0])

    img_list = return_picture_list(img_dir)

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

    for img in img_list:
        image = load_image(width, height, img)

        if ann.TensorInfo.IsQuantized(input_tensor_info):
            image = np.uint8(image)
        else:
            image = np.float32(image/np.iinfo("uint8").max)

        input_tensors = ann.make_input_tensors([input_binding_info], [image])
        runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
        result = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict

    print("pyarmnn")

def openvino(model_dir, img_dir):
    model_dir = os.path.join(model_dir, "ov")
    print(model_dir)

    device_name = "CPU"

    check_directories(model_dir, img_dir)

    print(os.listdir(model_dir))

    for entry in os.listdir(model_dir):
        if ".xml" in entry:
            model_dir = os.path.join(model_dir, entry)

    print(model_dir)

    img_list = return_picture_list(img_dir)

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


    for img in img_list:
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
        results = compiled_model.infer_new_request({0: input_tensor})

# --------------------------- Step 7. Process output ------------------------------------------------------------------
        predictions = next(iter(results.values()))

        # Change a shape of a numpy.ndarray with results to get another one with one dimension
        probs = predictions.reshape(-1)

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
    
    print("openvino")

def onnx_runtime(model_dir, img_dir):
    model_dir = os.path.join(model_dir, "onnx")
    print(model_dir)

    check_directories(model_dir, img_dir)
    model_dir = os.path.join(model_dir, os.listdir(model_dir)[0])
    img_list = return_picture_list(img_dir)

    session= onnxruntime.InferenceSession(model_dir)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image_height = session.get_inputs()[0].shape[2]
    image_width = session.get_inputs()[0].shape[3]

    for img in img_list:
        output = session.run([output_name], {input_name:preprocess_image(img, image_height, image_width)})[0]
        output = output.flatten()
        output = softmax(output) # this is optional
        top5_catid = np.argsort(-output)[:5]
        print(top5_catid)


    print("onnx")

def pytorch(model_dir, img_dir, class_dir):
    model_dir = os.path.join(model_dir, "pytorch")
    print(model_dir)

    check_directories(model_dir, img_dir)
    #model_dir = os.path.join(model_dir, os.listdir(model_dir)[0])
    img_list = return_picture_list(img_dir)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


    for img in img_list:
        input_image = Image.open(img)

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) 

        with torch.no_grad():
            output = model(input_batch)

        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        get_result(class_dir, probabilities)


        # Read the categories


    print("Pytorch")




def main():
    print("Main")

    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module')

    parser.add_argument("-api", '--api', help='inference API', required=False)
    parser.add_argument("-mdl", '--model', help='model', required=False)
    parser.add_argument("-cl", "--classes", help="txt file with classes", required=False)
    #parser.add_argument("-img", "--images", help="images for inference", required=False)

    args = parser.parse_args()

    general_dir = os.path.abspath(os.path.dirname(__file__)).split("scripts")[0]
    img_dir = os.path.join(general_dir, "images")
    model_dir = os.path.join(general_dir, "models")
    model_dir = os.path.join(model_dir, args.model)
    class_dir = os.path.join(general_dir, "classes")
    class_dir = os.path.join(class_dir, args.classes)

    print(model_dir)

    if args.api == "tflite_runtime":
        tflite_runtime(model_dir, img_dir, class_dir)
    elif args.api == "pyarmnn":
        pyarmnn(model_dir, img_dir, class_dir)
    elif args.api == "onnx":
        onnx_runtime(model_dir, img_dir, class_dir)
    elif args.api == "ov":
        openvino(model_dir, img_dir, class_dir)
    elif args.api == "pytorch":
        pytorch(model_dir, img_dir, class_dir)




  


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

