import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
#model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
#model.eval()
for param in model.parameters():
    param.requires_grad_(True)
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0).clone().requires_grad_(True) # create a mini-batch as expected by the model


# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

#with torch.no_grad():
output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        inputs=input_batch,
        outputs=output[0][top5_catid[0]-1],
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(output[0][top5_catid[0]]),
        create_graph=True,
        retain_graph=True,
    )
heatmap = gradient[0].squeeze(0)
heatmap = np.array(heatmap.detach().to('cpu'))
#heatmap = np.maximum(heatmap, 0)
heatmap = (heatmap + 1)/2
heatmap /= np.max(heatmap)
heatmap *= 255
heatmap = np.reshape(heatmap,(heatmap.shape[1],heatmap.shape[2],3))


import cv2
heatmap = cv2.resize(heatmap, (1213, 1546))
heatmap = heatmap.astype(np.uint8)
cv2.imshow('video', heatmap)
cv2.waitKey(5)

#image = np.maximum(np.array(heatmap.permute(2,1,0).detach().to('cpu')),0)