import numpy as np
import cv2
import pyarmnn as ann

def load_image(height, width, image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (height, width))
    cv2.imwrite("resized_input.jpg", resized_img)
    resized_img = np.expand_dims(resized_img, axis=0)
    return img, resized_img

def draw_rect_and_return_result(output_data, height, width, img, resized_img, score_threshold, labels):

    print(output_data[0])
    output_data = output_data[0][0]               # x(1, 25200, 7) to x(25200, 7)
    print(output_data.shape)
    print(output_data)
    
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze(output_data[..., 4:5])
    classes = classFilter(output_data[..., 5:])
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

    resized_img = resized_img[0,:,:]
    resized_img = resized_img *255
    #cv2.imwrite("before_img.jpg", resized_img)
    print(resized_img.shape)
    print()
    print(classes)
    print(len(scores))
    print()

    status = cv2.imwrite("before_img.jpg", resized_img)
    print(len(classes), len(scores))
    for i in range(len(scores)):
        if scores[i] > 0.9:
            print(i)
            print(scores[i])
            print(classes[i] + 1)
            print(labels[classes[i] + 1])
            
            x_min = int(max(1,(xyxy[0][i] * width)))
            y_min = int(max(1,(xyxy[1][i] * height)))
            x_max = int(min(height,(xyxy[2][i] * width)))
            y_max = int(min(width,(xyxy[3][i] * height)))

            
            resized_img = cv2.rectangle(resized_img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
            print(resized_img.shape)
            status = cv2.imwrite("img.jpg", resized_img)
            resized_img = cv2.putText(resized_img, labels[classes[i] + 1], (x_min+2, y_max-2),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            out_img = cv2.resize(resized_img, (img.shape[1], img.shape[0]))
            status = cv2.imwrite("img2.jpg", out_img)
            print(status)
    
    return None, None

def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

model_path = "yolov5l-fp16.tflite"
image_path = "dog.jpg"
labels_path = "coco.names"
score_threshold = 0.7

with open(labels_path, "r") as f:
    labels = [s.strip() for s in f.readlines()]

print(f"Working with ARMNN {ann.ARMNN_VERSION}")

parser = ann.ITfLiteParser()
network = parser.CreateNetworkFromBinaryFile(model_path)

options = ann.CreationOptions()
runtime = ann.IRuntime(options)


preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef'), ann.BackendId('GpuAcc')]
opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

print(f"Preferred Backends: {[back.Get() for back in preferredBackends]}\n {runtime.GetDeviceSpec()}\n")
print(f"Optimizationon warnings: {messages}")

# get input binding information for the input layer of the model
graph_id = parser.GetSubgraphCount() - 1
input_names = parser.GetSubgraphInputTensorNames(graph_id)
input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
input_tensor_id = input_binding_info[0]
input_tensor_info = input_binding_info[1]
width, height = input_tensor_info.GetShape()[1], input_tensor_info.GetShape()[2]
print(f"tensor id: {input_tensor_id},tensor info: {input_tensor_info}")

# Get output binding information for an output layer by using the layer name.
output_names = parser.GetSubgraphOutputTensorNames(graph_id)

output_binding_info = []

for output_name in output_names:
    output_binding_info.append(parser.GetNetworkOutputBindingInfo(graph_id, output_name))
output_tensors = ann.make_output_tensors(output_binding_info)

net_id, _ = runtime.LoadNetwork(opt_network)

image, resized_image = load_image(width, height, image_path)

if ann.TensorInfo.IsQuantized(input_tensor_info):
    resized_image = np.uint8(resized_image)
else:
    resized_image = np.float32(resized_image/np.iinfo("uint8").max)

input_tensors = ann.make_input_tensors([input_binding_info], [resized_image])
runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
output_data = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict
result, out_image = draw_rect_and_return_result(output_data, width, height, image, resized_image, score_threshold, labels)
out_path = "output_pyarmnn_det.jpg"
cv2.imwrite(out_path, out_image)