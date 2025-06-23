import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

from openvino.runtime import Core

# IoU function (same as before)
def compute_iou(pred_mask, true_mask, num_classes=21):
    ious = []
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        true_inds = (true_mask == cls)
        intersection = (pred_inds & true_inds).sum()
        union = (pred_inds | true_inds).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

# Load PyTorch model for baseline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pt = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).to(device)
model_pt.eval()

# OpenVINO setup
ie = Core()
model_path = "deeplabv3_mobilenetv3_large.xml"  # your OpenVINO IR .xml file
compiled_model = ie.compile_model(model_path, device_name="CPU")  # or "GPU" if available
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Dataset & preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=True, transform=transform)

num_samples = 50
miou_pt_list = []
miou_ov_list = []

for idx in range(num_samples):
    img, target = dataset[idx]
    input_tensor = img.unsqueeze(0).to(device)

    # PyTorch inference
    with torch.no_grad():
        output = model_pt(input_tensor)['out']
        pred_pt = output.argmax(1).squeeze().cpu().numpy()

    # Prepare input for OpenVINO: numpy float32, shape [1,C,H,W]
    input_data = img.unsqueeze(0).numpy()

    # OpenVINO inference
    result = compiled_model([input_data])[output_layer]  # shape: [1, 21, 512, 512]
    pred_ov = np.argmax(result[0], axis=0)

    # Resize ground truth mask to prediction size
    target_pil = Image.fromarray(np.array(target))
    target_resized = target_pil.resize(pred_ov.shape[::-1], resample=Image.NEAREST)
    target_np = np.array(target_resized)
    target_np[target_np == 255] = 0

    # Compute mIoU
    miou_pt = compute_iou(pred_pt, target_np, num_classes=21)
    miou_ov = compute_iou(pred_ov, target_np, num_classes=21)

    miou_pt_list.append(miou_pt)
    miou_ov_list.append(miou_ov)

    print(f"Sample {idx+1}/{num_samples} - PyTorch mIoU: {miou_pt:.4f}, OpenVINO mIoU: {miou_ov:.4f}")

print(f"\nAverage PyTorch mIoU: {np.nanmean(miou_pt_list):.4f}")
print(f"Average OpenVINO mIoU: {np.nanmean(miou_ov_list):.4f}")
