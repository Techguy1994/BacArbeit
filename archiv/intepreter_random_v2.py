import argparse
import io
import time
import numpy as np
import os

from PIL import Image
from tflite_runtime.interpreter import Interpreter

def write_txt_file(model_dict):
  file = open("intepreter_avg_time.txt", "w")

  for model, avg_time in model_dict.items():
    file.write(f"{model}, {avg_time}\n")

  file.close()



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
  
  length = 50

  dir_path = os.path.dirname(os.path.abspath(__file__))
  files_in_path = sorted(os.listdir(dir_path))
  models = [p for p in files_in_path if ".tflite" in p]
  #print(models)
  #print(dir_path)

  result_dict = {}

  for model in models:
    model_path = os.path.join(dir_path, model)

    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    image = np.random.randint(0,255, size=(height,width,3))

    tot_time = 0

    for i in range(length):
      if interpreter.get_input_details()[0]['dtype'] == np.float32:
        image = image/255
      start_int_time = time.time()
      results = classify_image(interpreter, image)
      tot_time = tot_time + (time.time() - start_int_time)
      label_id, prob = results[0]
      print(f"Time: {time.time() - start_int_time}")

    print(f"Tot time: {tot_time}, Avg time: {(tot_time)/length}")
    result_dict.update({model: tot_time/length})
  print(result_dict)
  write_txt_file(result_dict)


if __name__ == '__main__':
  main()
