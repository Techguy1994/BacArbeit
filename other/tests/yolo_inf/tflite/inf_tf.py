from wave import Wave_read
import numpy as np
#from PIL import Image
import cv2
import tensorflow as tf
import sys

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

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_h, input_w = input_shape[1], input_shape[2]
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
image_data = cv2.resize(image_data, (input_h, input_w))
resized_img = image_data
image_data = np.float32(image_data / 255)
image_data = np.expand_dims(image_data, axis=0)

#inference 
interpreter.set_tensor(input_details[0]['index'], image_data)
interpreter.invoke()
#output_data = interpreter.get_tensor(output_details[0]['index'])

output_data = []

for det in output_details:
    output_data.append(interpreter.get_tensor(det['index']))    

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
    if ((scores[i] > 1.0) or (scores[i] < 0.0)):
        sys.exit("out of bounds")
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