import argparse
import io
import time
import numpy as np
import os

from PIL import Image
from tflite_runtime.interpreter import Interpreter

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

  
def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

def main():

  # getting required files
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model', help='File path of .tflite file.', required=True)
  parser.add_argument('--labels', help='File path of labels file.', required=True)
  parser.add_argument('--pics', help='File path of pics folder.', required=True)
  args = parser.parse_args()

  labels = load_labels(args.labels) 
  path = args.pics
  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  #print(interpreter.get_input_details()[0]['dtype'])
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  start_tot_time = time.time()

  for file in os.listdir(path):
    start_int_time = time.time()
    image = os.path.join(path, file)
    image = Image.open(image).convert('RGB').resize((width, height), Image.ANTIALIAS)
    if interpreter.get_input_details()[0]['dtype'] == np.float32:
      image = np.array(image)/255
    results = classify_image(interpreter, image)
    label_id, prob = results[0]
    end_int_time = time.time()
    print(f"Index: {label_id}, pred: {labels[label_id]} with {prob*100}%")
    print(f"Time: {end_int_time - start_int_time}")

  end_tot_time = time.time()
  print(f"Tot time: {end_tot_time-start_tot_time}, Avg time: {(end_tot_time-start_tot_time)/len(os.listdir(path))}")


if __name__ == '__main__':
  main()
