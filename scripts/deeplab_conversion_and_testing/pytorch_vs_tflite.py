import torch
import torchvision
import numpy as np
import tensorflow as tf
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

# --- IoU computation ---
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

# --- Load PyTorch DeepLabV3 MobileNetV3 Large ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pt = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).to(device)
model_pt.eval()

# --- Load TFLite model ---
tflite_model_path = 'deeplabv3_mobilenetv3_large_float32.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Dataset preprocessing ---
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=True, transform=transform)

num_samples = 50
miou_pt_list = []
miou_tflite_list = []

for idx in range(num_samples):
    img, target = dataset[idx]
    input_tensor = img.unsqueeze(0).to(device)

    # PyTorch inference
    with torch.no_grad():
        output = model_pt(input_tensor)['out']
        pred_pt = output.argmax(1).squeeze().cpu().numpy()

    # Prepare input for TFLite: NHWC float32
    input_data = np.array(img.permute(1, 2, 0))  # CHW to HWC
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Handle output shape
    if output_data.shape[-1] == 21:
        pred_tflite = np.argmax(output_data[0], axis=-1)
    else:
        pred_tflite = np.argmax(output_data[0].transpose(1, 2, 0), axis=-1)

    # Resize ground truth mask to match prediction size
    target_pil = Image.fromarray(np.array(target))
    target_resized = target_pil.resize(pred_pt.shape[::-1], resample=Image.NEAREST)
    target_np = np.array(target_resized)
    target_np[target_np == 255] = 0  # ignore label replaced

    # Compute mIoU
    miou_pt = compute_iou(pred_pt, target_np, num_classes=21)
    miou_tflite = compute_iou(pred_tflite, target_np, num_classes=21)

    miou_pt_list.append(miou_pt)
    miou_tflite_list.append(miou_tflite)

    print(f"Sample {idx+1}/{num_samples} - PyTorch mIoU: {miou_pt:.4f}, TFLite mIoU: {miou_tflite:.4f}")

print(f"\nAverage PyTorch mIoU: {np.nanmean(miou_pt_list):.4f}")
print(f"Average TFLite mIoU: {np.nanmean(miou_tflite_list):.4f}")
