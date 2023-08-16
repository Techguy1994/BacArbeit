
def preprocess_tflite_moobilenet(image_path, height, width, data_type):
    from PIL import Image
    import numpy as np

    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS)
    image_data = np.asarray(image).astype(data_type)
    if data_type is np.float32:
        for channel in range(image_data.shape[0]):
            image_data[channel, :, :] = (image_data[channel, :, :] / 127.5) - 1
    image_data = np.expand_dims(image_data, 0)
    return image_data

def preprocess_onnx_mobilenet(image_path, height, width, data_type):
    from PIL import Image
    import numpy as np

    if "float" in data_type:
        type = np.float32
    else:
        type = np.uint8

    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS)
    image_data = np.asarray(image).astype(type)
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = image_data[channel, :, :] / 255 - mean[channel] / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

def preprocess_tflite_yolov5(image_path, height, width, data_type):
    import numpy as np
    #import cv2

    image_data = cv2.imread(image_path)
    orig_img = image_data
    image_data = cv2.resize(image_data, (height, width))
    #resized_img = image_data
    if data_type is np.float32:
        print("float model")
        image_data = np.float32(image_data / 255)
    image_data = np.expand_dims(image_data, axis=0)
    print(image_data.shape)

    return image_data, orig_img

def preprocess_onnx_yolov5(image_path, input_type, height, width):
    import numpy as np
    import cv2
    image_data = cv2.imread(image_path)
    #orig_img = image_data
    image_data = cv2.resize(image_data, (height, width))
    #resized_img = image_data
    #if input_type is np.float32:
    #    print("float model")
    print("before: ", image_data.shape)
    image_data = image_data.transpose([2, 0, 1])
    print("after: ", image_data.shape)
    image_data = np.float32(image_data / 255)
    image_data = np.expand_dims(image_data, axis=0)
    print(image_data.shape)

    return image_data