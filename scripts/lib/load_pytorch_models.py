    # sample execution (requires torchvision)
from torchvision import transforms, models
import torch


def load_pytorch_model(name):

    if name == "mobilenet_v2":
        model = "models." + name + "(pretrained=True)"
        print(model)
    return model


