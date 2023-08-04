def run_tf(args):
    import tensorflow as tf
    import time 
    import lib.postprocess as post
    import lib.preprocess as pre
    from time import perf_counter
    import lib.data as dat
    import pandas as pd

    
    output_dict = dat.create_base_dictionary_class(args.n_big)

    interpreter = tf.lite.Interpreter(model_path=args.model, experimental_delegates=None, num_threads=4)

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
                output_data = interpreter.get_tensor(output_details[0]['index'])
                output = post.handle_output_tf(output_data, output_details, args.label, args.n_big)
                output_dict = dat.store_output_dictionary_class(output_dict, image, lat, output, args.n_big)
                df = dat.create_pandas_dataframe(output_dict)
            else:
                interpreter.invoke()
                lat = 0
                output_data = interpreter.get_tensor(output_details[0]['index'])
                output = post.handle_output_tf(output_data, output_details, args.label, args.n_big)
                output_dict = dat.store_output_dictionary_class(output_dict, image, lat, output, args.n_big)
                df = dat.create_pandas_dataframe(output_dict)

        time.sleep(args.sleep)

    return df

def run_pyarmnn(args):
    import pyarmnn as ann
    import numpy as np
    import time 
    import lib.postprocess as post
    import lib.preprocess as pre
    from time import perf_counter
    import lib.data as dat
    import pandas as pd

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
                result = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
                output = post.handle_output_pyarmnn(result, args.label, args.n_big)
                output_dict = dat.store_output_dictionary_class(output_dict, image, lat, output, args.n_big)
                df = dat.create_pandas_dataframe(output_dict)
            else:
                lat = 0
                runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
                result = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
                output = post.handle_output_pyarmnn(result, args.label, args.n_big)
                output_dict = dat.store_output_dictionary_class(output_dict, image, lat, output, args.n_big)
                df = dat.create_pandas_dataframe(output_dict)

        time.sleep(args.sleep)

    return df

def run_onnx(args):
    import onnxruntime
    import time 
    import lib.postprocess as post
    import lib.preprocess as pre
    from time import perf_counter
    import lib.data as dat
    import pandas as pd

    output_dict = dat.create_base_dictionary_class(args.n_big)

    options = onnxruntime.SessionOptions()

    #, 'XNNPACKExecutionProvider'
    providers = ['CPUExecutionProvider']

    print("optimize")
    options.intra_op_num_threads = 4
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    #options.inter_op_num_threads = 4
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
            processed_image = pre.preprocess_onnx_mobilenet(image, image_height, image_width, input_data_type)

            if args.profiler == "perfcounter":
                start_time = perf_counter()
                result = session.run([output_name], {input_name:processed_image})[0]
                end_time = perf_counter()
                lat = end_time - start_time
                print("time in ms: ", lat*1000)
                output = post.handle_output_onnx(result, output_data_type, args.label, args.n_big)
                output_dict = dat.store_output_dictionary_class(output_dict, image, lat, output, args.n_big)
                df = dat.create_pandas_dataframe(output_dict)

        time.sleep(args.sleep)

    return df



