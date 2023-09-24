
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

def preprocess_pytorch_mobilenet():
    from torchvision import transforms

    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    return preprocess

def preprocess_tflite_yolov5(image_path, height, width, data_type):
    import numpy as np
    import cv2

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

def preprocess_tf_deeplab(image, height, width, input_type, channels=3):
    import cv2
    import numpy as np
    #image = Image.open(image_path)
    #image = image.resize((height, width), Image.LANCZOS)
    #image_data = np.asarray(image).astype(input_type)
    #for channel in range(image_data.shape[0]):
    #    image_data[channel, :, :] = image_data[channel, :, :]*127.5 - 1
    #image_data = np.expand_dims(image_data, 0)
    #print(image_data.shape)
    #quit()
    #return image, image_data
    

    image = cv2.imread(image)
    image = cv2.resize(image, (width, height))
    image_data = cv2.normalize(image.astype(input_type), None, -1.0, 1.0, cv2.NORM_MINMAX)
    image_data = np.expand_dims(image_data, 0)

    return image, image_data

def preprocess_onnx_deeplab(image, input_type, image_height, image_width):
    import cv2
    import numpy as np

    image = cv2.resize(image, (image_height, image_width))
    image_data = cv2.normalize(image.astype(np.float32), None, -1.0, 1.0, cv2.NORM_MINMAX)
    image_data = np.expand_dims(image_data, 0)

    return image_data

