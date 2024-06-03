
def preprocess_tflite_moobilenet(image_path, height, width, data_type):
    from PIL import Image
    import numpy as np

    image = Image.open(image_path)
    
    image = image.resize((width, height), Image.LANCZOS)
    #image = image.convert("RGB")
    #image_data = np.asarray(image).astype(data_type)


    if len(np.shape(np.asarray(image).astype(data_type))) == 2:
        image = image.convert("RGB")
    
    image_data = np.asarray(image).astype(data_type)

    #print(image_data.flags)
    #image_data = np.atleast_3d(image)
    #image_data.setflags(write=1)
  

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
    #image_data = np.asarray(image).astype(type)


    if len(np.shape(np.asarray(image).astype(type))) == 2:
        #image_data = np.expand_dims(image_data, 2)
        image = image.convert("RGB")

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


    return image_data, orig_img

def preprocess_onnx_yolov5(image_data, input_type, height, width):
    import numpy as np
    import cv2

    image_data = cv2.resize(image_data, (height, width))
    image_data = image_data.transpose([2, 0, 1])
    #image_data = np.float32(image_data)
    image_data = np.float32(image_data/255)
    image_data = np.expand_dims(image_data, axis=0)

    return image_data

def preprocess_pytorch_yolo():
    import torch
    from torchvision import models, transforms
    """
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    """
    preprocess = transforms.Compose([transforms.ToTensor()])

    return preprocess

def preprocess_tf_deeplab(image, input_shape, input_type):
    import cv2
    import numpy as np
    import sys

    image = cv2.imread(image)
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image_data = cv2.normalize(image.astype(input_type), None, 0, 1.0, cv2.NORM_MINMAX)
    image_data = np.expand_dims(image_data, 0)

    return image_data


def preprocess_tf_deeplab_alt(image, input_shape, input_type):
    import cv2
    import numpy as np
    import sys


    image = cv2.imread(image)
    image = cv2.resize(image, (input_shape[2], input_shape[3]))
    image = image.transpose([2,0,1])

    
    image_data = cv2.normalize(image.astype(np.float32), None, 0, 1.0, cv2.NORM_MINMAX)
    image_data = np.expand_dims(image_data, 0)

    print(image_data.shape)

    return image_data

def preprocess_onnx_deeplab(image, input_type, image_height, image_width):
    import cv2
    import numpy as np
    import sys

    image = cv2.imread(image)
    image = cv2.resize(image, (image_height, image_width))

    #image_data = cv2.normalize(image.astype(np.float32), None, -1.0, 1.0, cv2.NORM_MINMAX)
    image_data = cv2.normalize(image.astype(np.float32), None, 0, 1.0, cv2.NORM_MINMAX)
    #image_data = image.transpose([2,0,1])

    image_data = np.expand_dims(image_data, 0)

    return image_data

def preprocess_onnx_deeplab_alt(image, input_type, image_height, image_width):
    import cv2
    import numpy as np
    import sys

    image = cv2.imread(image)
    print(image.shape)
    image = cv2.resize(image, (image_height, image_width))
    print(image.shape)
    image = image.transpose([2,0,1])
    print(image.shape)
    #sys.exit()
    #image_data = cv2.normalize(image.astype(np.float32), None, -1.0, 1.0, cv2.NORM_MINMAX)
    image_data = cv2.normalize(image.astype(np.float32), None, 0, 1.0, cv2.NORM_MINMAX)
    #image_data = image.transpose([2,0,1])
    image_data = np.expand_dims(image_data, 0)
    print(image_data.shape)


    return image_data

def preprocess_pytorch_seg():

    from torchvision import models, transforms

    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    return preprocess

def preprocess_pytorch_deeplab(image, preprocess):

    from torchvision import transforms

    from PIL import Image
    input_image = Image.open(image)
    input_image = input_image.convert("RGB")

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def preprocess_ov_yolo(shape, image):
    import cv2
    import numpy as np

        # --------------------------- Step 3. Set up input --------------------------------------------------------------------
    # Read input images
    #images = [cv2.imread(image_path) for image_path in args.images]

    # Resize images to model input dims
    _, _, h, w = shape
    #_, h, w, _ = model.input().shape
    #print("Model input shape: ",model.input().shape)
    #h, w = 224, 224

    #resized_images = [cv2.resize(image, (w, h)) for image in images]
    resized_image = cv2.resize(image, (w, h))

    # Add N dimension
    #input_tensors = [np.expand_dims(image, 0) for image in resized_images]
    input_tensor = np.expand_dims(resized_image, 0)
    #print("input tensor shape: ", input_tensors[0].shape)

    return input_tensor

def preprocess_ov_mobilenet(shape, image):
    import cv2
    import numpy as np

    #print("---")
    #print(shape)
        # --------------------------- Step 3. Set up input --------------------------------------------------------------------
    # Read input images
    #images = [cv2.imread(image_path) for image_path in args.images]

    # Resize images to model input dims
    _, h, w, _ = shape
    #_, h, w, _ = model.input().shape
    #print("Model input shape: ",model.input().shape)
    #h, w = 224, 224
    #print(h,w)

    #resized_images = [cv2.resize(image, (w, h)) for image in images]
    resized_image = cv2.resize(image, (w, h))

    # Add N dimension
    #input_tensors = [np.expand_dims(image, 0) for image in resized_images]
    input_tensor = np.expand_dims(resized_image, 0)

    #print("shape of tensor: ", input_tensor.shape)
    #print("input tensor shape: ", input_tensors[0].shape)

    return input_tensor



