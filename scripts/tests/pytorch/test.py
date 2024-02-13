
import argparse
import logging as log
import os
import sys
from time import sleep, perf_counter
import cProfile, pstats
import numpy as np
import cv2
from PIL import Image

from openvino.runtime import InferRequest

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(1, 3, 224, 224)


with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))