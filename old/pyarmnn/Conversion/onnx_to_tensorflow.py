# source: https://github.com/onnx/onnx-tensorflow/blob/master/example/onnx_to_tf.py

import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("ImageClassifier.onnx")  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph("tf_class")  # export the model