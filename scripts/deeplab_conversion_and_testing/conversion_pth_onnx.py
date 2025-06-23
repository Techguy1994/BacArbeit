import torch
import torchvision

model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 512, 512)  # Batch size 1, 3 channels, 512x512 image

torch.onnx.export(
    model,
    dummy_input,
    "deeplabv3_mobilenetv3_large.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
)

