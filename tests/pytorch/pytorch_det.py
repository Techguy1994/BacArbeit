import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Image
img = 'https://ultralytics.com/images/zidane.jpg'

# Inference
results = model(img)
print("results:", results)
#print("hey", results.xyxy[0].shape)
#print("hey 2", results.xyxy[0])

output = results.xyxy[0]

print(output.shape)
print(output)
print("\n")

print(output[:,0:4])
print(output[:,4])
print(output[:,5])


#results.pandas().xyxy[0]
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie