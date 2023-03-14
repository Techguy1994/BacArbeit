import argparse
import logging
import os
from time import sleep, time
import numpy as np
import pyarmnn as ann
import tflite_runtime.interpreter as tflite
import cv2

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

    print("openvino")

def onnx_runtime(model_dir, img_dir):
    model_dir = os.path.join(model_dir, "onnx")
    print(model_dir)

    print("onnx")

def pytorch(model_dir, img_dir):
    model_dir = os.path.join(model_dir, "pytorch")
    print(model_dir)

    print("Pytorch")




def main():
    print("Main")

    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module')

    parser.add_argument("-api", '--api', help='inference API', required=False)
    parser.add_argument("-mdl", '--model', help='model', required=False)
    #parser.add_argument("-img", "--images", help="images for inference", required=False)

    args = parser.parse_args()

    general_dir = os.path.abspath(os.path.dirname(__file__)).split("scripts")[0]
    img_dir = os.path.join(general_dir, "images")
    model_dir = os.path.join(general_dir, "models")
    model_dir = os.path.join(model_dir, args.model)
    print(model_dir)

    if args.api == "tflite_runtime":
        tflite_runtime(model_dir, img_dir)
    elif args.api == "pyarmnn":
        pyarmnn(model_dir, img_dir)
    elif args.api == "onnx":
        onnx_runtime(model_dir, img_dir)
    elif args.api == "ov":
        openvino(model_dir, img_dir)
    elif args.api == "pytorch":
        pytorch(model_dir, img_dir)




  


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

