
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
        #print("float")
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

    #print(output_details[0]["dtype"])

    #if "integer" in output_details:
    #    print("int")
    #    quit("no adapted to onnx, please change following code when quantized model is given")
    #    out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    #elif "float" in output_details:
    #    print("float")
    #    out_normalization_factor = 1

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = probs[entry] / out_normalization_factor
        #result[entry] = [val*100]
        #print("\tpos {} : {:.2f}%".format(entry, val*100))
        results.append({"label": label[entry], "index:": entry, "value": val})
        
    return results


def handle_output_tf_yolo_det(output_details, intepreter, original_image, thres, file_name, label):
    import numpy as np
    import cv2

    print(thres)

    results = []
    output_data = []

    for det in output_details:
        output_data.append(intepreter.get_tensor(det['index']))



    output_data = output_data[0][0]
    print(output_data.shape)

    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    orig_W, orig_H = original_image.shape[1], original_image.shape[0]
    print("Boxes shape: ", boxes.shape)
    print("scores shape: ", scores.shape)
    print("Classes Len", len(classes))
    print("Orig: ", original_image.shape)
    print(orig_H, orig_W)

    output_img = original_image

    for i in range(len(scores)):
        if ((scores[i] > thres) and (scores[i] <= 1.0)):
            print(label[classes[i]],classes[i], scores[i])
            xmin = int(max(1,(xyxy[0][i] * orig_W)))
            ymin = int(max(1,(xyxy[1][i] * orig_H)))
            xmax = int(min(orig_W,(xyxy[2][i] * orig_W)))
            ymax = int(min(orig_H,(xyxy[3][i] * orig_H)))

            output_img = cv2.rectangle(output_img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            results.append({"label": label[classes[i]],"index": classes[i], "value": scores[i]})

        
    print(output_img.shape)
    cv2.imwrite(file_name, output_img)

    return results

def handle_output_pyarmnn_yolo_det(output_details, img_org, thres, img_result_file, label):
    import cv2
    import numpy as np
    print("Output pyarmnn")

    results = []
    output_data = output_details[0][0]
    #print(len(output_details))
    print(output_data.shape)
    #output_data = output_data[0][0]

    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    orig_W, orig_H = img_org.shape[1], img_org.shape[0]
    print("Boxes shape: ", boxes.shape)
    print("scores shape: ", scores.shape)
    print("Classes Len", len(classes))
    print("Orig: ", img_org.shape)
    print(orig_H, orig_W)

    output_img = img_org

    for i in range(len(scores)):
        if ((scores[i] > thres) and (scores[i] <= 1.0)):
            #print(labels[classes[i]],classes[i], scores[i])
            xmin = int(max(1,(xyxy[0][i] * orig_W)))
            ymin = int(max(1,(xyxy[1][i] * orig_H)))
            xmax = int(min(orig_W,(xyxy[2][i] * orig_W)))
            ymax = int(min(orig_H,(xyxy[3][i] * orig_H)))

            output_img = cv2.rectangle(output_img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            results.append({"label": label[classes[i]],"index": classes[i], "value": scores[i]})

        
    print(output_img.shape)
    cv2.imwrite(img_result_file, output_img)

    return results

def handle_output_onnx_yolo_det(output_details, img_org, thres, img_result_file, label, model_shape):
    import numpy as np
    import cv2
    results = []

    print(output_details)
    output_data = output_details[0]
    print(output_data.shape)
    output_data = output_data[0]

    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    orig_W, orig_H = img_org.shape[1], img_org.shape[0]
    print("Boxes shape: ", boxes.shape)
    print("scores shape: ", scores.shape)
    print("Classes Len", len(classes))
    print("Orig: ", img_org.shape)
    print(orig_H, orig_W)

    ratio_H, ratio_W = orig_H/model_shape[0], orig_W/model_shape[1]
    print(ratio_H, ratio_W)

    output_img = img_org

    for i in range(len(scores)):
        if ((scores[i] > thres) and (scores[i] <= 1.0)):
            #print(labels[classes[i]],classes[i], scores[i])
            #print(xyxy[0][i], xyxy[1][i], xyxy[2][i], xyxy[3][i])
            xmin, ymin, xmax, ymax = int(xyxy[0][i]*ratio_W), int(xyxy[1][i]*ratio_H), int(xyxy[2][i]*ratio_W), int(xyxy[3][i]*ratio_H)
            #xmin = int(max(1,(xyxy[0][i] * orig_W)))
            #ymin = int(max(1,(xyxy[1][i] * orig_H)))
            #xmax = int(min(orig_W,(xyxy[2][i] * orig_W)))
            #ymax = int(min(orig_H,(xyxy[3][i] * orig_H)))

            #print(xmin, xmax, ymin, ymax)

            output_img = cv2.rectangle(output_img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            #cv2.imwrite("test.jpg", output_img)
            #sys.exit()
            results.append({"label": label[classes[i]],"index": classes[i], "value": scores[i]})

        
    print(output_img.shape)
    cv2.imwrite(img_result_file, output_img)

    return results

def handle_output_pytorch_yolo_det(output_details, img_org, thres, img_result_file, label, model_shape):

    import cv2

    results = []



    print(output_details)
    output_data = output_details[0]
    print(output_data.shape)
    #output_data = output_data[0]
    print(output_data.shape)

    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    orig_W, orig_H = img_org.shape[1], img_org.shape[0]
    print("Boxes shape: ", boxes.shape)
    print("scores shape: ", scores.shape)
    print("Classes Len", len(classes))
    print("Orig: ", img_org.shape)
    print(orig_H, orig_W)

    #ratio_H, ratio_W = orig_H/576, orig_W/768
    #print(ratio_H, ratio_W)

    output_img = img_org

    for i in range(len(scores)):
        if ((scores[i] > thres) and (scores[i] <= 1.0)):
            print(label[classes[i]],classes[i], scores[i])
            print(xyxy[0][i], xyxy[1][i], xyxy[2][i], xyxy[3][i])

            xmin, ymin, xmax, ymax = int(xyxy[0][i]), int(xyxy[1][i]), int(xyxy[2][i]), int(xyxy[3][i])
            #sys.exit()
            #xmin = int(max(1,(xyxy[0][i] * orig_W)))
            #ymin = int(max(1,(xyxy[1][i] * orig_H)))
            #xmax = int(min(orig_W,(xyxy[2][i] * orig_W)))
            #ymax = int(min(orig_H,(xyxy[3][i] * orig_H)))

            print(xmin, xmax, ymin, ymax)

            output_img = cv2.rectangle(output_img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            #cv2.imwrite("test.jpg", output_img)
            #sys.exit()
            results.append({"label": label[classes[i]],"index": classes[i], "value": scores[i]})

        
    print(output_img.shape)
    cv2.imwrite(img_result_file, output_img)

    return results

def handle_output_deeplab_tf(output_details, interpreter, image, raw_file, overlay_file, colormap, label):
    import numpy as np
    import cv2

    #print(output_details[0])
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # my method, first resize, then argmax 
    output_data = output_data[0]

    #print(output_data.shape)

    seg_map = np.ndarray((image.shape[0],image.shape[1],len(output_data[0,0,:])))

    for i in range(len(output_data[0,0,:])):
        seg_map[:,:,i] = cv2.resize(output_data[:,:,i], (image.shape[1], image.shape[0]))


    seg_map = np.argmax(seg_map, axis=2)

    result, out_image, out_mask = vis_segmentation_cv2(image, seg_map, label, colormap)
    #print(img_result_file)
    #results.append(result)
    #gen_out_path = os.path.join(img_result_file, img_res.split("/")[-1].split(".")[0])
    #mask_out_path = gen_out_path + "_mask.jpg"
    #result_pic_out_path = gen_out_path + ".jpg"
    cv2.imwrite(overlay_file, out_image)
    cv2.imwrite(raw_file, out_mask)

    return result

def handle_output_deeplab_pyarmnn(output_data, image, raw_file, overlay_file, colormap, label):
    import cv2
    import numpy as np

    #print("label shape", labels.shape)

    #print(output_data[0].shape)
    #print(output_data[0][0].shape)
    #print(output_data[0][0])
    
    output_data = output_data[0][0]

    #print(output_data[0,0,:])
    #print(len(output_data[0,0,:]))
    #print("output_data", output_data.shape)


    #seg_map = np.ndarray((img_res.shape[0],img_res.shape[1],len(output_data[0,0,:])))
    seg_map = np.ndarray((image.shape[0],image.shape[1],len(output_data[0,0,:])))

    #print("segmap", seg_map.shape)

    for i in range(len(output_data[0,0,:])):
        seg_map[:,:,i] = cv2.resize(output_data[:,:,i], (image.shape[1],image.shape[0]))


    seg_map = np.argmax(seg_map, axis=2)

    #print("segmap after argmax (labels)", seg_map.shape)

    result, out_image, out_mask = vis_segmentation_cv2(image, seg_map, label, colormap)
    #print(img_result_file)
    #results.append(result)
    #gen_out_path = os.path.join(img_result_file, img_res.split("/")[-1].split(".")[0])
    #mask_out_path = gen_out_path + "_mask.jpg"
    #result_pic_out_path = gen_out_path + ".jpg"
    cv2.imwrite(overlay_file, out_image)
    cv2.imwrite(raw_file, out_mask)

    return result

def handle_output_deeplab_onnx(output_data, origanal_image, raw_file, overlay_file, colormap, label):
    import numpy as np
    import cv2

    output_data = output_data[0]
    print(len(output_data))
    print(len(output_data[0,0,:]))

    seg_map = np.ndarray((origanal_image.shape[0],origanal_image.shape[1],len(output_data[0,0,:])))

    for i in range(len(output_data[0,0,:])):
        seg_map[:,:,i] = cv2.resize(output_data[:,:,i], (origanal_image.shape[1],origanal_image.shape[0]))


    seg_map = np.argmax(seg_map, axis=2)

    result, out_image, out_mask = vis_segmentation_cv2(origanal_image, seg_map, label, colormap)
    #print(img_result_file)
    #results.append(result)
    #gen_out_path = os.path.join(img_result_file, img_res.split("/")[-1].split(".")[0])
    #mask_out_path = gen_out_path + "_mask.jpg"
    #result_pic_out_path = gen_out_path + ".jpg"
    cv2.imwrite(overlay_file, out_image)
    cv2.imwrite(raw_file, out_mask)

    return result

def handle_output_deeplab_pytorch(output_data, image, raw_file, overlay_file, colormap, label):
    import numpy as np
    import cv2

    #https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd-mobilenetv1
    results = []



    output = output_data.numpy()
    
    
    #print(output_details[0])
   # output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # my method, first resize, then argmax 
    #output_data = output_data[0]

    #print(output_data.shape)
    #print(output.shape)
    #print(img_res.shape)
    #print(len(output_data[:,0,0]))

    seg_map = np.ndarray((image.shape[0],image.shape[1],len(output[:,0,0])))

    #print(seg_map.shape)

    for i in range(len(output[:,0,0])):
        seg_map[:,:,i] = cv2.resize(output[i,:,:], (image.shape[1],image.shape[0]))


    seg_map = np.argmax(seg_map, axis=2)

    result, out_image, out_mask = vis_segmentation_cv2(image, seg_map, label, colormap)
    #print(img_result_file)
    #results.append(result)
    #gen_out_path = os.path.join(img_result_file, img_res.split("/")[-1].split(".")[0])
    #mask_out_path = gen_out_path + "_mask.jpg"
    #result_pic_out_path = gen_out_path + ".jpg"
    cv2.imwrite(overlay_file, out_image)
    cv2.imwrite(raw_file, out_mask)

    return result


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

    #print(image.shape, )
    cv2.imwrite("result1.jpg", image)
    seg_image = label_to_color_image(seg_map, colormap).astype(np.uint8)

    #print(image.shape, seg_image.shape)
    #print(LABEL_NAMES)

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
    #print(unique_labels) 
    #print(indeces)
    #print(res)

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

    #print("Label shape: ", label.shape)  

    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    #print("label to color")
    #print(label)
    #print(colormap)
    #print(label.shape, colormap.shape)
    #print(colormap[label])

    return colormap[label]