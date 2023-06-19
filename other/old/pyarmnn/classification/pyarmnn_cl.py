import pyarmnn as ann
import numpy as np
import os
import csv
import cv2

def load_image(height, width):
    
    img = cv2.imread("test.jpg")
    img = cv2.resize(img, (height, width))
    cv2.imwrite("Out.jpg", img)
    #print()
    #print(img)
    #print()
    img = np.expand_dims(img, axis=0)
    return img

def return_n_biggest_result(output_data, n_big=3):

    max_positions = np.argpartition(output_data[0][0], -n_big)[-n_big:]
    out_normalization_factor = np.iinfo(output_data[0].dtype).max
    
    result = []

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[0][0][entry] / out_normalization_factor
        result.append(val*100)
        print("\tpos {} : {:.2f}%".format(entry, val*100))
        
    print()

    return result

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

def write_txt_file(model_dict):
  file = open("pyarmnn_avg_time.txt", "w")

  for model, avg_time in model_dict.items():
    file.write(f"{model}, {avg_time}\n")

  file.close()

def setup_profiling(net_id, runtime):
    profiler = runtime.GetProfiler(net_id)
    profiler.EnableProfiling(True)
    return profiler

def print_profiling_data(profiler):
    profiler_data = ann.get_profiling_data(profiler)

    times = profiler_data.inference_data["execution_time"]
    tot_time = 0
    for time in times:
        print(f"inference model time: {round(time/1000, 5)}ms")
        tot_time += time

    avg_time = tot_time / len(times)
 
    print(f"Total_time: {round(tot_time/1000,5)}ms, avg_time: {round(avg_time/1000, 5)}ms")

def write_profiling_data(profiler, model_path):
    profiler_data = ann.get_profiling_data(profiler)

    csv_path = "/home/ubuntu2104/pyarmnn/inf_times"
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




def inf_pyarmnn(model_path, iterations=50):
    # LINK TO CODE: https://www.youtube.com/watch?v=HQYosuy4ABY&t=1867s
    #https://developer.arm.com/documentation/102557/latest
    #file:///C:/Users/Maroun_Desktop_PC/SynologyDrive/Bachelorarbeit/pyarmnn/pyarmnn_doc.html#pyarmnn.IOutputSlot

    print(f"Working with ARMNN {ann.ARMNN_VERSION}")

    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile(model_path)

    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef'), ann.BackendId('GpuAcc')]
    opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    print(f"Preferred Backends: {preferredBackends}\n {runtime.GetDeviceSpec()}\n")
    print(f"Optimizationon warnings: {messages}")

    graph_id = parser.GetSubgraphCount() - 1
    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
    input_tensor_id = input_binding_info[0]
    input_tensor_info = input_binding_info[1]
    width, height = input_tensor_info.GetShape()[1], input_tensor_info.GetShape()[2]
    print(f"tensor id: {input_tensor_id},tensor info: {input_tensor_info}")

    #image = np.random.randint(0,255, size=(height,width,3))
    #image = np.array(np.random.random_sample((height,width,3)), dtype=np.uint8)
    

    # Get output binding information for an output layer by using the layer name.
    output_names = parser.GetSubgraphOutputTensorNames(graph_id)

    output_binding_info = []

    for output_name in output_names:
        output_binding_info.append(parser.GetNetworkOutputBindingInfo(graph_id, output_name))
    output_tensors = ann.make_output_tensors(output_binding_info)

    net_id, _ = runtime.LoadNetwork(opt_network)

    # Setup the Profilier for layer and network and inference time 
    profiler = setup_profiling(net_id, runtime)

    results = []

    for i in range(iterations):
        image = load_image(width, height)

        if ann.TensorInfo.IsQuantized(input_tensor_info):
            image = np.uint8(image)
        else:
            image = np.float32(image/255)

        input_tensors = ann.make_input_tensors([input_binding_info], [image])
        runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
        result = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
        result = return_n_biggest_result(result, 3)
        results.append(result)

        
    print(results)

    # Profiler Data 
    print_profiling_data(profiler)
    write_profiling_data(profiler, model_path)


if __name__ == "__main__":
    iterations = 1

    dir_path = os.path.dirname(os.path.abspath(__file__))
    files_in_path = sorted(os.listdir(os.path.join(dir_path, "models")))
    models = [p for p in files_in_path if ".tflite" in p]
    print(models)

    try: 
        for i, model in enumerate(models):
            model_path = os.path.join(dir_path, "models", model)
            inf_pyarmnn(model_path, iterations=iterations)
    except KeyboardInterrupt:
        print("Keyboard interrupt")



