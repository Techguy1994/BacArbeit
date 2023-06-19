#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys
from time import perf_counter

import cv2
import numpy as np
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type


def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    # Parsing and validation of input arguments
    #if len(sys.argv) != 4:
    #    log.info(f'Usage: {sys.argv[0]} <path_to_model> <path_to_image> <device_name>')
    #    return 1

    model_path = "yolov5l.onnx"
    image_path = "dog.jpg"
    device_name = "CPU"
    class_dir = "coco.names"
    thres = 0.7
    values = []

    with open(class_dir, "r") as f:
        labels = [s.strip() for s in f.readlines()]

# --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

# --------------------------- Step 2. Read a model --------------------------------------------------------------------
    log.info(f'Reading the model: {model_path}')
    # (.xml and .bin files) or (.onnx file)
    model = core.read_model(model_path)

    print(model.inputs, model.outputs)
    

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1

    if len(model.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

# --------------------------- Step 3. Set up input --------------------------------------------------------------------
    # Read input image
    image = cv2.imread(image_path)

    print(model.input().shape)
    _, _, h, w = model.input().shape

    resized_image = cv2.resize(image, (w,h))

    transposed_image = resized_image.transpose([2, 0, 1])

    scaled_image = transposed_image / 255

    # Add N dimension
    input_tensor = np.expand_dims(scaled_image, 0)

    #sys.exit(input_tensor.shape)

# --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
    ppp = PrePostProcessor(model)

    _, h, w, _ = input_tensor.shape

    # 1) Set input tensor information:
    # - input() provides information about a single model input
    # - reuse precision and shape from already available `input_tensor`
    # - layout of data is 'NHWC'
    #ppp.input().tensor() \
    #    .set_shape(input_tensor.shape) \
    #    .set_element_type(Type.u8) \
    #    .set_layout(Layout('NHWC'))  # noqa: ECE001, N400

    # 2) Adding explicit preprocessing steps:
    # - apply linear resize from tensor spatial dims to model spatial dims
    #ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)

    # 3) Here we suppose model has 'NCHW' layout for input
    #ppp.input().model().set_layout(Layout('NCHW'))

    # 4) Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    #ppp.output().tensor().set_element_type(Type.f32)

    # 5) Apply preprocessing modifying the original 'model'
    model = ppp.build()

# --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    log.info('Loading the model to the plugin')
    compiled_model = core.compile_model(model, device_name)

# --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
    log.info('Starting inference in synchronous mode')
    beg = perf_counter()
    results = compiled_model.infer_new_request({0: input_tensor})
    end = perf_counter()
    diff = end - beg
    print("Time:", diff*1000)

# --------------------------- Step 7. Process output ------------------------------------------------------------------
    print(results)
    for res in results:
        output = results[res]

    print(output.shape)
    output = output[0]
    print(output.shape)

    boxes = np.squeeze(output[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    print(boxes.shape, scores.shape, len(classes))

    output_img = resized_image

    for i in range(len(scores)):
        if ((scores[i] > 1.0) or (scores[i] < 0)):
            sys.exit("irregular score")
        if ((scores[i] > thres) and (scores[i] <= 1.0)):
            print(labels[classes[i]],classes[i], scores[i])
            #print(xyxy[0][i], xyxy[1][i], xyxy[2][i], xyxy[3][i])
            xmin, ymin, xmax, ymax = int(xyxy[0][i]), int(xyxy[1][i]), int(xyxy[2][i]), int(xyxy[3][i])
            #xmin = int(max(1,(xyxy[0][i] * orig_W)))
            #ymin = int(max(1,(xyxy[1][i] * orig_H)))
            #xmax = int(min(orig_W,(xyxy[2][i] * orig_W)))
            #ymax = int(min(orig_H,(xyxy[3][i] * orig_H)))

            #print(xmin, xmax, ymin, ymax)

            output_img = cv2.rectangle(output_img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            #cv2.imwrite("test.jpg", output_img)
            #sys.exit()
            values.append({"object": labels[classes[i]],"index": classes[i], "accuracy": scores[i]})

        
    print(output_img.shape)
    cv2.imwrite("output.jpg", output_img)


    #predictions = next(iter(results.values()))

    # Change a shape of a numpy.ndarray with results to get another one with one dimension
    #probs = predictions.reshape(-1)

    # Get an array of 10 class IDs in descending order of probability
    #top_10 = np.argsort(probs)[-10:][::-1]

    #header = 'class_id probability'

    #log.info(f'Image path: {image_path}')
    #log.info('Top 10 results: ')
    #log.info(header)
    #log.info('-' * len(header))

    #for class_id in top_10:
    #    probability_indent = ' ' * (len('class_id') - len(str(class_id)) + 1)
    #    log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}')

    log.info('')

# ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
