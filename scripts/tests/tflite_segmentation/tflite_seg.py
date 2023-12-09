import numpy as np
import tensorflow as tf
import cv2
import sys


if len(sys.argv) < 4:
    print("Usage: " + sys.argv[0] + " model_path image_path colormap_path")
    quit()

model_path = sys.argv[1]
image_path = sys.argv[2]
colormap_path = sys.argv[3]

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
img = cv2.imread(image_path)
img = cv2.resize(img, (input_shape[1], input_shape[2]))
input_data = cv2.normalize(img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
#print(input_data.shape)

input_data = tf.reshape(input_data, input_shape)
#print(input_data.shape)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
#print(output_data.shape)
output_classes = np.uint8(tf.argmax(output_data, axis=3)[0])
#print(output_classes.shape)
output_classes_rgb = cv2.cvtColor(output_classes, cv2.COLOR_GRAY2RGB)
print(output_classes_rgb.shape)
#print(output_classes_rgb)
cv2.imwrite("test_image_rgb.png", output_classes_rgb)
colormap = cv2.imread(colormap_path).astype(np.uint8)
print(colormap.shape)
output_img = cv2.LUT(output_classes_rgb, colormap)

cv2.imwrite("test_image.png", output_img)

resized_image = cv2.resize(output_img, (500,281), interpolation = cv2.INTER_AREA)
cv2.imwrite("resized_test_image.png", resized_image)
