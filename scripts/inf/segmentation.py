def run_tf(args):
    print("todo")

def run_pyarmnn():
    print("todo")

def run_onnx(args, output_image_folder, raw_folder, overlay_folder):
    print("Chosen API: Onnx runtime")

    import onnxruntime
    import time 
    import lib.postprocess as post
    import lib.preprocess as pre
    from time import perf_counter
    import lib.data as dat
    import pandas as pd
    import cv2

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

            if args.profiler == "perfcounter":
                processed_image = pre.preprocess_onnx_deeplab(original_image, input_data_type, image_height, image_width)

                start_time = perf_counter()
                result = session.run(outputs, {input_name:processed_image})[0]
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)

def run_pytorch():
    print("todo")

def run_sync_openvino():
    print("todo")

def run_async_openvino():
    print("todo")