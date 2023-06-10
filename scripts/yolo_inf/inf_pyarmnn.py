import numpy as np
import cv2
import pyarmnn as ann

def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

model_path = "yolov5l-fp16.tflite"
image_path = "dog.jpg"
#image_path = "zidane.jpg"
#labels_path = "labelmap_changed_coco_detection.txt"
labels_path = "coco.names"

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
#
# print(input_shape)

# Preprocess the image with Image
"""
image_data = Image.open(image_path)
orig_img = image_data
image_data = image_data.resize((640,640))
image_data = image_data / 255
image_data = np.array(image_data).astype(np.float32)
resized_img = image_data
image_data = np.expand_dims(image_data, axis=0)
"""

#Preprocess image with cv2
image_data = cv2.imread(image_path)
orig_img = image_data
image_data = cv2.resize(image_data, (height, width))
resized_img = image_data
image_data = np.float32(image_data / 255)
image_data = np.expand_dims(image_data, axis=0)
print("input image shape: ", image_data.shape)

input_tensors = ann.make_input_tensors([input_binding_info], [image_data])
runtime.EnqueueWorkload(0, input_tensors, output_tensors) # inference call
output_data = ann.workload_tensors_to_ndarray(output_tensors) # gather inference results into dict




#output processing
#print(output_data.shape)
output_data = output_data[0][0]
print(output_data.shape)

boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
classes = classFilter(output_data[..., 5:]) # get classes
# Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

orig_W, orig_H = orig_img.shape[1], orig_img.shape[0]
print("Boxes shape: ", boxes.shape)
print("scores shape: ", scores.shape)
print("Classes Len", len(classes))
print("Orig: ", orig_img.shape)
print(orig_H, orig_W)
#print("res: ", resized_img.shape)

#W, H = resized_img.shape[1], resized_img.shape[0]


#print(W,H)

output_img = orig_img

for i in range(len(scores)):
    if ((scores[i] > 0.7) and (scores[i] <= 1.0)):
        print(labels[classes[i]],classes[i], scores[i])
        #H = frame.shape[0]
        #W = frame.shape[1]
        xmin = int(max(1,(xyxy[0][i] * orig_W)))
        ymin = int(max(1,(xyxy[1][i] * orig_H)))
        xmax = int(min(orig_W,(xyxy[2][i] * orig_W)))
        ymax = int(min(orig_H,(xyxy[3][i] * orig_H)))

        output_img = cv2.rectangle(output_img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
    
#output_img = cv2.resize(resized_img, (orig_W, orig_H))
print(output_img.shape)
cv2.imwrite("output.jpg", output_img)