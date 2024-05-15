def handle_output_tf(output_data, output_details, label, n_big):
    import numpy as np
    results = []

    max_positions = np.argpartition(output_data[0], -n_big)[-n_big:]
    #print(output_details[0]["dtype"])

    if output_details[0]['dtype'] == np.uint8:
        out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    elif output_details[0]['dtype'] == np.float32:
        out_normalization_factor = 1

    

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[0][entry] / out_normalization_factor
        results.append({"label":  label[entry], "index": entry, "value": val})

    return results

def handle_output_pyarmnn(output_data, label, n_big):
    import numpy as np
    results = []

    output_data = output_data[0]
    max_positions = np.argpartition(output_data[0], -n_big)[-n_big:]

    if output_data.dtype == "uint8":
        out_normalization_factor = np.iinfo(output_data.dtype).max
    elif output_data.dtype == "float32":
        out_normalization_factor = 1
    
    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[0][entry] / out_normalization_factor
        results.append({"label": label[entry], "index": entry, "value": val})
        
    return results

def handle_output_onnx_mobilenet_class(output_data, output_details, label, n_big):
    import numpy as np
    results = []

    output_data = output_data.flatten()
    output_data = softmax(output_data) # this is optional

    max_positions = np.argpartition(output_data, -n_big)[-n_big:]
    out_normalization_factor = 1

    #print(output_details[0]["dtype"])

    if "integer" in output_details:
        print("int")
        quit("not adapted to onnx, please change following code when quantized model is given")
        out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    elif "float" in output_details:
        out_normalization_factor = 1


    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[entry] / out_normalization_factor
        results.append({"label":  label[entry], "index": entry, "value": val})
        
    return results

def handle_output_pytorch_mobilenet_class(output_data, label, n_big):
    import numpy as np
    import torch

    results = []

    probabilities = torch.nn.functional.softmax(output_data[0], dim=0)

    probabilities = probabilities.detach().numpy()

    #prob = probabilities.item()
    #print(torch.sum(probabilities))

    max_positions = np.argpartition(probabilities, -n_big)[-n_big:]


    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = probabilities[entry] 

        results.append({"label": label[entry], "index": entry.item(), "value": val.item()})
        
    return results

def handle_output_openvino_moiblenet_class(output_data, label, n_big):
    import numpy as np

    results = []

    predictions = next(iter(output_data.values()))
    probs = predictions.reshape(-1)

    max_positions = np.argpartition(probs, -n_big)[-n_big:]
    out_normalization_factor = 1



    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = probs[entry] / out_normalization_factor
        results.append({"label": label[entry], "index": entry, "value": val})
        
    return results


def handle_output_tf_yolo_det_old(output_details, intepreter, original_image, thres, file_name, label):
    import numpy as np
    import cv2

    results = []
    output_data = []

    for det in output_details:
        output_data.append(intepreter.get_tensor(det['index']))



    output_data = output_data[0][0]

    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    orig_W, orig_H = original_image.shape[1], original_image.shape[0]


    output_img = original_image

    for i in range(len(scores)):
        if ((scores[i] > thres) and (scores[i] <= 1.0)):
            xmin = int(max(1,(xyxy[0][i] * orig_W)))
            ymin = int(max(1,(xyxy[1][i] * orig_H)))
            xmax = int(min(orig_W,(xyxy[2][i] * orig_W)))
            ymax = int(min(orig_H,(xyxy[3][i] * orig_H)))

            output_img = cv2.rectangle(output_img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            results.append({"label": label[classes[i]],"index": classes[i], "value": scores[i]})

    cv2.imwrite(file_name, output_img)

    return results

def handle_output_tf_yolo_det(output_details, intepreter, original_image, thres, file_name, label):
    import numpy as np
    import cv2
    import sys

    results = []
    output_data = []
    all_det = []
    nms_det = []


    for det in output_details:
        output_data.append(intepreter.get_tensor(det['index']))

    output_data = output_data[0][0]

    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh

    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    orig_W, orig_H = original_image.shape[1], original_image.shape[0]

    output_img = original_image
    for i in range(len(scores)):
        if ((scores[i] > thres) and (scores[i] <= 1.0)):

            xmin = max(1,(xyxy[0][i] * orig_W))
            ymin = max(1,(xyxy[1][i] * orig_H))
            xmax = min(orig_W,(xyxy[2][i] * orig_W))
            ymax = min(orig_H,(xyxy[3][i] * orig_H))

            all_det.append((classes[i],[xmin, ymin, xmax, ymax], scores[i]))



    while all_det:
        element = int(np.argmax([all_det[i][2] for i in range(len(all_det))]))
        nms_det.append(all_det.pop(element))
        all_det = [*filter(lambda x: (iou(x[1], nms_det[-1][1]) <= 0.45), [det for det in all_det])]

    for det in nms_det:

        x_left = round(det[1][0], 2)
        y_left = round(det[1][1], 2)
        w = round(det[1][2] - det[1][0], 2)
        h = round(det[1][3] - det[1][1], 2)
  
        output_img = cv2.rectangle(output_img, (int(det[1][0]), int(det[1][1])), (int(det[1][2]), int(det[1][3])), (10, 255, 0), 2)
        results.append({"label": label[det[0]],"index": det[0], "value": det[2], "boxes": [x_left, y_left, w, h]})

    cv2.imwrite(file_name, output_img)

    return results

def handle_output_tf_yolo_det_alt(output_details, intepreter, original_image, thres, file_name, label):
    import numpy as np
    import cv2
    import sys

    results = []
    output_data = []
    all_det = []
    nms_det = []


    for det in output_details:
        output_data.append(intepreter.get_tensor(det['index']))



    output_data = output_data[0][0]


    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh

    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]
    coco_xywh = [x- w/2,y-h/2,w,h]

    orig_W, orig_H = original_image.shape[1], original_image.shape[0]


    output_img = original_image

    for i in range(len(scores)):
        if ((scores[i] > thres) and (scores[i] <= 1.0)):

            xl = coco_xywh[0][i]
            yl = coco_xywh[1][i]
            wl = coco_xywh[2][i]
            hl = coco_xywh[3][i]

            all_det.append((classes[i],[xl, yl, wl, hl], scores[i]))


    while all_det:
        element = int(np.argmax([all_det[i][2] for i in range(len(all_det))]))
        nms_det.append(all_det.pop(element))
        all_det = [*filter(lambda x: (iou(x[1], nms_det[-1][1], orig_W, orig_H) <= 0.4), [det for det in all_det])]

    

    for det in nms_det:

        x1 = det[1][0] 
        y1 = det[1][1] 
        w1 = det[1][2]
        h1 = det[1][3]

        x_min_org = x1
        y_min_org = y1
        x_max_org = x1 + w1
        y_max_org = y1 + h1

        xmin = max(1,(x_min_org * orig_W))
        ymin = max(1,(y_min_org * orig_H))
        xmax = min(orig_W,(x_max_org * orig_W))
        ymax = min(orig_H,(y_max_org * orig_H))

        x_orig = round(x1 * orig_H, 2)
        y_orig = round(y1 * orig_H, 2)
        w_orig = round(w1 * orig_W, 2)
        h_orig = round(h1 * orig_H, 2)

        bbox = [x_orig, y_orig, w_orig, h_orig]

        output_img = cv2.rectangle(output_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (10, 255, 0), 2)
        results.append({"label": label[det[0]],"index": det[0], "value": det[2], "boxes": bbox})

    cv2.imwrite(file_name, output_img)

    return results

def handle_output_pyarmnn_yolo_det(output_details, img_org, thres, img_result_file, label):
    import cv2
    import numpy as np

    results = []
    output_data = output_details[0][0]


    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    orig_W, orig_H = img_org.shape[1], img_org.shape[0]


    output_img = img_org

    for i in range(len(scores)):
        if ((scores[i] > thres) and (scores[i] <= 1.0)):
            
            xmin = int(max(1,(xyxy[0][i] * orig_W)))
            ymin = int(max(1,(xyxy[1][i] * orig_H)))
            xmax = int(min(orig_W,(xyxy[2][i] * orig_W)))
            ymax = int(min(orig_H,(xyxy[3][i] * orig_H)))

            output_img = cv2.rectangle(output_img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            results.append({"label": label[classes[i]],"index": classes[i], "value": scores[i]})

        
    
    cv2.imwrite(img_result_file, output_img)

    return results

def handle_output_onnx_yolo_det(output_details, img_org, thres, img_result_file, label, model_shape):
    import numpy as np
    import cv2
    import sys

    results = []
    all_det = []
    nms_det = []

    output_data = output_details[0]
    output_data = output_data[0]

    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    orig_W, orig_H = img_org.shape[1], img_org.shape[0]


    ratio_H, ratio_W = orig_H/model_shape[0], orig_W/model_shape[1]

    output_img = img_org

    for i in range(len(scores)):
        if ((scores[i] > thres) and (scores[i] <= 1.0)):

            xmin = max(1,xyxy[0][i])
            ymin = max(1,xyxy[1][i])
            xmax = min(model_shape[0], xyxy[2][i])
            ymax = min(model_shape[0],xyxy[3][i])

            all_det.append((classes[i],[xmin, ymin, xmax, ymax], scores[i]))
            


    while all_det:
        element = int(np.argmax([all_det[i][2] for i in range(len(all_det))]))
        nms_det.append(all_det.pop(element))
        all_det = [*filter(lambda x: (iou(x[1], nms_det[-1][1]) <= 0.45), [det for det in all_det])]

        
    for det in nms_det:
        xmin = det[1][0]*ratio_W
        ymin = det[1][1]*ratio_H
        xmax = det[1][2]*ratio_W
        ymax = det[1][3]*ratio_H

        #output_img = cv2.rectangle(output_img, (int(det[1][0]),int(det[1][1])), (int(det[1][2]), int(det[1][3])), (10, 255, 0), 2)
        output_img = cv2.rectangle(output_img, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (10, 255, 0), 2)

        x_left = round(xmin, 2)
        y_left = round(ymin, 2)
        w = round(xmax - xmin, 2)
        h = round(ymax - ymin, 2)

        results.append({"label": label[det[0]],"index": det[0], "value": det[2], "boxes": [x_left,y_left, w, h]})

    cv2.imwrite(img_result_file, output_img)

    return results

def handle_output_pytorch_yolo_det(output, img_org, thres, img_result_file, label, model_shape):

    import cv2
    import sys
    import numpy as np

    results = []
    all_det = []
    nms_det = []

    output = output.xyxy[0]
    output = output.numpy()

    # format xmin, ymin, xmax, ymax, score, index (label)

    for element in output:
        
        if ((element[4] > thres) and (element[4] <= 1.0)):
            all_det.append((int(element[5]), [element[0], element[1], element[2], element[3]], element[4]))

    
    while all_det:
        element = int(np.argmax([all_det[i][2] for i in range(len(all_det))]))
        nms_det.append(all_det.pop(element))
        all_det = [*filter(lambda x: (iou(x[1], nms_det[-1][1]) <= 0.45), [det for det in all_det])]

        
    for det in nms_det:
        output_img = cv2.rectangle(img_org, (int(det[1][0]),int(det[1][1])), (int(det[1][2]), int(det[1][3])), (10, 255, 0), 2)


        #convert to x,y,w,h
        x_left = round(det[1][0], 2)
        y_left = round(det[1][1], 2)
        w = round(det[1][2] - det[1][0], 2)
        h = round(det[1][3] - det[1][1], 2)


        results.append({"label": label[det[0]],"index": det[0], "value": det[2], "boxes": [x_left,y_left,w, h]})

    if results:
        cv2.imwrite(img_result_file, output_img)

    return results

def handle_output_ov_yolo_det(output_details, img_org, thres, img_result_file, label, model_shape):
    import cv2 
    import numpy as np
    import sys

    results = []
    all_det = []
    nms_det = []

    for elem in output_details:
        output_data = output_details[elem]


    output_data = output_data[0]


    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    orig_W, orig_H = img_org.shape[1], img_org.shape[0]


    ratio_H, ratio_W = orig_H/model_shape[0], orig_W/model_shape[1]


    output_img = img_org

    for i in range(len(scores)):
        if ((scores[i] > thres) and (scores[i] <= 1.0)):
            #xmin, ymin, xmax, ymax = xyxy[0][i]*ratio_W, xyxy[1][i]*ratio_H, xyxy[2][i]*ratio_W, xyxy[3][i]*ratio_H
            xmin = max(1,xyxy[0][i])
            ymin = max(1,xyxy[1][i])
            xmax = min(model_shape[0],xyxy[2][i])
            ymax = min(model_shape[0],xyxy[3][i])

            all_det.append((classes[i],[xmin, ymin, xmax, ymax], scores[i]))

        
    while all_det:
        element = int(np.argmax([all_det[i][2] for i in range(len(all_det))]))
        nms_det.append(all_det.pop(element))
        all_det = [*filter(lambda x: (iou(x[1], nms_det[-1][1]) <= 0.45), [det for det in all_det])]

        
    for det in nms_det:
        xmin = det[1][0]*ratio_W
        ymin = det[1][1]*ratio_H
        xmax = det[1][2]*ratio_W
        ymax = det[1][3]*ratio_H

        x_left = round(xmin, 2)
        y_left = round(ymin, 2)
        w = round(xmax - xmin, 2)
        h = round(ymax - ymin, 2)

        output_img = cv2.rectangle(output_img, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (10, 255, 0), 2)
        results.append({"label": label[det[0]],"index": det[0], "value": det[2], "boxes": [x_left,y_left, w, h]})

    cv2.imwrite(img_result_file, output_img)

    return results
        


def handle_output_deeplab_tf(output_details, interpreter, image, raw_file, overlay_file, colormap, label):
    import numpy as np
    import cv2

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # my method, first resize, then argmax 
    output_data = output_data[0]

    seg_map = np.ndarray((image.shape[0],image.shape[1],len(output_data[0,0,:])))

    for i in range(len(output_data[0,0,:])):
        seg_map[:,:,i] = cv2.resize(output_data[:,:,i], (image.shape[1], image.shape[0]))


    seg_map = np.argmax(seg_map, axis=2)

    result, out_image, out_mask = vis_segmentation_cv2(image, seg_map, label, colormap)

    cv2.imwrite(overlay_file, out_image)
    cv2.imwrite(raw_file, out_mask)

    return result

def handle_output_deeplab_tf_alt(output_details, interpreter, image, raw_file, overlay_file, index_file, colormap, label):
    import numpy as np
    import cv2
    import sys

    width = image.shape[1]
    height = image.shape[0]

    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_classes = np.uint8(np.argmax(output_data, axis=3)[0])


    

    unique_labels = np.unique(output_classes)
    label = np.asarray(label)
    results = label[unique_labels]

    output_image = np.zeros((output_classes.shape[0], output_classes.shape[1], 3), dtype=np.uint8)

    #colormap given in RGB but apparently BGR needed

    for i in range(output_classes.shape[0]):
        for j in range(output_classes.shape[1]):
            output_image[i][j][0] = colormap[output_classes[i][j]][2]
            output_image[i][j][1] = colormap[output_classes[i][j]][1]
            output_image[i][j][2] = colormap[output_classes[i][j]][0]


    resized_image = cv2.resize(output_image, (width,height), interpolation = cv2.INTER_AREA)
    index_image = cv2.resize(output_classes, (width, height), interpolation = cv2.INTER_AREA)
    overlay_image = cv2.addWeighted(image, 0.7, resized_image, 0.5, 0)

    cv2.imwrite(overlay_file, overlay_image)
    cv2.imwrite(raw_file, resized_image)
    cv2.imwrite(index_file, index_image)

    return results

def handle_output_deeplab_pyarmnn(output_data, image, raw_file, overlay_file, index_file, colormap, label):
    import cv2
    import numpy as np

    width = image.shape[1]
    height = image.shape[0]

    output_data = output_data[0]

    output_classes = np.uint8(np.argmax(output_data, axis=3)[0])

    unique_labels = np.unique(output_classes)
    label = np.asarray(label)
    results = label[unique_labels]

    output_image = np.zeros((output_classes.shape[0], output_classes.shape[1], 3), dtype=np.uint8)

    #colormap given in RGB but apparently BGR needed

    for i in range(output_classes.shape[0]):
        for j in range(output_classes.shape[1]):
            output_image[i][j][0] = colormap[output_classes[i][j]][2]
            output_image[i][j][1] = colormap[output_classes[i][j]][1]
            output_image[i][j][2] = colormap[output_classes[i][j]][0]


    resized_image = cv2.resize(output_image, (width,height), interpolation = cv2.INTER_AREA)
    index_image = cv2.resize(output_classes, (width, height), interpolation = cv2.INTER_AREA)
    overlay_image = cv2.addWeighted(image, 0.7, resized_image, 0.5, 0)

    cv2.imwrite(overlay_file, overlay_image)
    cv2.imwrite(raw_file, resized_image)
    cv2.imwrite(index_file, index_image)

    return results

def handle_output_deeplab_pyarmnn_old(output_data, image, raw_file, overlay_file, colormap, label):
    import cv2
    import numpy as np

    
    output_data = output_data[0][0]

    seg_map = np.ndarray((image.shape[0],image.shape[1],len(output_data[0,0,:])))



    for i in range(len(output_data[0,0,:])):
        seg_map[:,:,i] = cv2.resize(output_data[:,:,i], (image.shape[1],image.shape[0]))


    seg_map = np.argmax(seg_map, axis=2)


    result, out_image, out_mask = vis_segmentation_cv2(image, seg_map, label, colormap)
    print(result.shape)

    cv2.imwrite(overlay_file, out_image)
    cv2.imwrite(raw_file, out_mask)

    return result

def handle_output_deeplab_onnx(output_data, image, raw_file, overlay_file, index_file, colormap, label):
    import numpy as np
    import cv2
    import sys

    width = image.shape[1]
    height = image.shape[0]

    output_classes = np.uint8(np.argmax(output_data, axis=3)[0])

    unique_labels = np.unique(output_classes)
    label = np.asarray(label)
    results = label[unique_labels]

    output_image = np.zeros((output_classes.shape[0], output_classes.shape[1], 3), dtype=np.uint8)

    #colormap given in RGB but apparently BGR needed

    for i in range(output_classes.shape[0]):
        for j in range(output_classes.shape[1]):
            output_image[i][j][0] = colormap[output_classes[i][j]][2]
            output_image[i][j][1] = colormap[output_classes[i][j]][1]
            output_image[i][j][2] = colormap[output_classes[i][j]][0]


    resized_image = cv2.resize(output_image, (width,height), interpolation = cv2.INTER_AREA)
    index_image = cv2.resize(output_classes, (width, height), interpolation = cv2.INTER_AREA)
    overlay_image = cv2.addWeighted(image, 0.7, resized_image, 0.5, 0)

    cv2.imwrite(overlay_file, overlay_image)
    cv2.imwrite(raw_file, resized_image)
    cv2.imwrite(index_file, index_image)

    return results

def handle_output_deeplab_onnx_old(output_data, origanal_image, raw_file, overlay_file, colormap, label):
    import numpy as np
    import cv2

    output_data = output_data[0]


    seg_map = np.ndarray((origanal_image.shape[0],origanal_image.shape[1],len(output_data[0,0,:])))

    for i in range(len(output_data[0,0,:])):
        seg_map[:,:,i] = cv2.resize(output_data[:,:,i], (origanal_image.shape[1],origanal_image.shape[0]))


    seg_map = np.argmax(seg_map, axis=2)

    result, out_image, out_mask = vis_segmentation_cv2(origanal_image, seg_map, label, colormap)

    cv2.imwrite(overlay_file, out_image)
    cv2.imwrite(raw_file, out_mask)

    return result

def handle_output_deeplab_pytorch(output_data, image, raw_file, overlay_file, index_file, colormap, label):
    import numpy as np
    import cv2
    import sys

    width = image.shape[1]
    height = image.shape[0]

    output_data = output_data.numpy()


    output_classes = np.uint8(np.argmax(output_data, axis=0))


    unique_labels = np.unique(output_classes)
    label = np.asarray(label)
    results = label[unique_labels]

    output_image = np.zeros((output_classes.shape[0], output_classes.shape[1], 3), dtype=np.uint8)

    #colormap given in RGB but apparently BGR needed

    for i in range(output_classes.shape[0]):
        for j in range(output_classes.shape[1]):
            output_image[i][j][0] = colormap[output_classes[i][j]][2]
            output_image[i][j][1] = colormap[output_classes[i][j]][1]
            output_image[i][j][2] = colormap[output_classes[i][j]][0]


    resized_image = cv2.resize(output_image, (width,height), interpolation = cv2.INTER_AREA)
    index_image = cv2.resize(output_classes, (width, height), interpolation = cv2.INTER_AREA)
    overlay_image = cv2.addWeighted(image, 0.7, resized_image, 0.5, 0)

    cv2.imwrite(overlay_file, overlay_image)
    cv2.imwrite(raw_file, resized_image)
    cv2.imwrite(index_file, index_image)

    return results

def handle_output_deeplab_pytorch_old(output_data, image, raw_file, overlay_file, colormap, label):
    import numpy as np
    import cv2

    #https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd-mobilenetv1

    output = output_data.numpy()
    
    seg_map = np.ndarray((image.shape[0],image.shape[1],len(output[:,0,0])))



    for i in range(len(output[:,0,0])):
        seg_map[:,:,i] = cv2.resize(output[i,:,:], (image.shape[1],image.shape[0]))


    seg_map = np.argmax(seg_map, axis=2)

    result, out_image, out_mask = vis_segmentation_cv2(image, seg_map, label, colormap)

    cv2.imwrite(overlay_file, out_image)
    cv2.imwrite(raw_file, out_mask)

    return result

def handle_output_deeplab_ov(output_data, image, raw_file, overlay_file, index_file, colormap, label):
    import numpy as np
    import cv2
    import sys

    width = image.shape[1]
    height = image.shape[0]

    for element in output_data:
        output = output_data[element][0]
        

    output_classes = np.uint8(np.array(output, dtype="int"))


    unique_labels = np.unique(output_classes)
    
    label = np.asarray(label)
    
    results = label[unique_labels]


    output_image = np.zeros((output_classes.shape[0], output_classes.shape[1], 3), dtype=np.uint8)
    

    #colormap given in RGB but apparently BGR needed

    for i in range(output_classes.shape[0]):
        for j in range(output_classes.shape[1]):
            output_image[i][j][0] = colormap[output_classes[i][j]][2]
            output_image[i][j][1] = colormap[output_classes[i][j]][1]
            output_image[i][j][2] = colormap[output_classes[i][j]][0]


    resized_image = cv2.resize(output_image, (width,height), interpolation = cv2.INTER_AREA)
    index_image = cv2.resize(output_classes, (width, height), interpolation = cv2.INTER_AREA)
    overlay_image = cv2.addWeighted(image, 0.7, resized_image, 0.5, 0)

    cv2.imwrite(overlay_file, overlay_image)
    cv2.imwrite(raw_file, resized_image)
    cv2.imwrite(index_file, index_image)

    return results


def softmax(x):
    import numpy as np
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def vis_segmentation_cv2(image, seg_map, LABEL_NAMES, colormap):
    import cv2
    import numpy as np
    """Visualizes input image, segmentation map and overlay view."""

   
    cv2.imwrite("result1.jpg", image)
    seg_image = label_to_color_image(seg_map, colormap).astype(np.uint8)


    overlay_picture = cv2.addWeighted(image, 0.7, seg_image, 0.5, 0)

    #LABEL_NAMES = np.asarray([
    #    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    #    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    #])

    #print(LABEL_NAMES)
    #FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    #FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP, colormap)


    unique_labels = np.unique(seg_map)

    LABEL_NAMES = np.asarray(LABEL_NAMES)
    #indeces = FULL_COLOR_MAP[unique_labels].astype(np.uint8)
    res = LABEL_NAMES[unique_labels]


    return res, overlay_picture, seg_image

def label_to_color_image(label, colormap):
    import numpy as np
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """


    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')


    return colormap[label]

def iou_old(box1_org, box2_org, orig_W, orig_H):
    import sys
    """
    Calculates the intersection-over-union (IoU) value for two bounding boxes.

    Args:
        box1: Array of positions for first bounding box
              in the form [x_min, y_min, x_max, y_max].
        box2: Array of positions for second bounding box.

    Returns:
        Calculated intersection-over-union (IoU) value for two bounding boxes.
    """



    x1 = box1_org[0] 
    y1 = box1_org[1] 
    w1 = box1_org[2]
    h1 = box1_org[3]

    x_min_org = x1
    y_min_org = y1
    x_max_org = x1 + w1
    y_max_org = y1 + h1

    xmin = max(1,(x_min_org * orig_W))
    ymin = max(1,(y_min_org * orig_H))
    xmax = min(orig_W,(x_max_org * orig_W))
    ymax = min(orig_H,(y_max_org * orig_H))

    box1 = [xmin, ymin, xmax, ymax]

    x1 = box2_org[0] 
    y1 = box2_org[1] 
    w1 = box2_org[2]
    h1 = box2_org[3]

    x_min_org = x1
    y_min_org = y1
    x_max_org = x1 + w1
    y_max_org = y1 + h1

    xmin = max(1,(x_min_org * orig_W))
    ymin = max(1,(y_min_org * orig_H))
    xmax = min(orig_W,(x_max_org * orig_W))
    ymax = min(orig_H,(y_max_org * orig_H))

    box2 = [xmin, ymin, xmax, ymax]

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area_box1 <= 0 or area_box2 <= 0:
        iou_value = 0
    else:
        y_min_intersection = max(box1[1], box2[1])
        x_min_intersection = max(box1[0], box2[0])
        y_max_intersection = min(box1[3], box2[3])
        x_max_intersection = min(box1[2], box2[2])

        area_intersection = max(0, y_max_intersection - y_min_intersection) *\
                            max(0, x_max_intersection - x_min_intersection)
        area_union = area_box1 + area_box2 - area_intersection

        try:
            iou_value = area_intersection / area_union
        except ZeroDivisionError:
            iou_value = 0

    return iou_value

def iou(box1, box2):
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area_box1 <= 0 or area_box2 <= 0:
        iou_value = 0
    else:
        y_min_intersection = max(box1[1], box2[1])
        x_min_intersection = max(box1[0], box2[0])
        y_max_intersection = min(box1[3], box2[3])
        x_max_intersection = min(box1[2], box2[2])

        area_intersection = max(0, y_max_intersection - y_min_intersection) *\
                            max(0, x_max_intersection - x_min_intersection)
        area_union = area_box1 + area_box2 - area_intersection

        try:
            iou_value = area_intersection / area_union
        except ZeroDivisionError:
            iou_value = 0

    return iou_value
