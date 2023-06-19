import torch.onnx
import os

def Convert_ONNX():
    pass


if __name__ == "__main__":


    #dir_path = os.path.dirname(os.path.abspath(__file__))
    #dir_path = os.path.join(dir_path, "models")
    #dir_path = os.path.join(dir_path, "pytorch")
    #dir_path = os.path.join(dir_path, "model.pth")
    #print(dir_path)
    #model = torch.load(dir_path, map_location=torch.device('cpu'))
    #model.eval()

    #soruces: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    #https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model

    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.eval()

    input_size = (224,224)

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1,3,224,224, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "ImageClassifier.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

