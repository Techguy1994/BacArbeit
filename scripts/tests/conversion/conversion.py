import torch
from torchvision import models, transforms
import torch.onnx 

#weights = models.MobileNet_V2_Weights('DEFAULT')

#model = models.quantization.mobilenet_v3_large(pretrained=True, quantize=True)
#

#model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_large', pretrained=True)

print(models.list_models())

#model = models.get_model("quantized_mobilenet_v3_large", weights="DEFAULT")
#model = models.get_model("mobilenet_v3_large", weights="DEFAULT")
#model = models.get_model("mobilenet_v3_small", weights="DEFAULT", quantize=True)
#model = models.get_model("mobilenet_v3_small", weights="DEFAULT", quantize=False)

model = models.get_model("mobilenet_v2", weights="DEFAULT")


#model = models.quantization.mobilenet_v2(weights=weights.DEFAULT, quantize=False)
#model = models.quantization.mobilenet_v2(pretrained=True, quantize=False)
#model = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
model.eval()

# Let's create a dummy input tensor  
dummy_input = torch.randn(1,3,224,224, requires_grad=True)  

torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "m_2.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 


print(" ") 
print('Model has been converted to ONNX') 