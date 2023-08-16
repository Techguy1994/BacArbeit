
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

def handle_output_onnx(output_data, output_details, label, n_big):
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

def handle_output_tf_det(output_details, intepreter, original_image, thres, file_name, label):
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