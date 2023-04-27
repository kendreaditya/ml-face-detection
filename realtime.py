# %%
import cv2
import numpy as np
import torch
import os 
from pathlib import Path
from torchvision.datasets import ImageFolder
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
from torchvision import transforms
from PIL import Image

# %%
from models import topKResnet18
#model = topKResnet18(None, k=64, columns=[f'{i}' for i in range(512)])
#Create video capture object

# %%
vid = cv2.VideoCapture(0)
arr = []
#preProcess_img = []
transform = transforms.Compose([
    # Data Preprocessing
    transforms.CenterCrop(224), # Crops the image to a 224 x 224 image 
    transforms.ToTensor(), 
    # Converts the Image Object to a Tensor object (which is how PyTorch keeps track of its gradients which is used for Backpropagation
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the images
])


cv2.namedWindow("Window")
count = 0
# %%
while(vid.isOpened()):

    #capture video frame by frame
    ret, frame = vid.read()
    #print(type(frame))
    cv2.imshow('Window', frame)
    cv2.imwrite(f'realtime_img\\frame%d.jpg' % count, frame)
    count+=1

    #quit the script using the q key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


#Release video capture object
vid.release()

#destroy all windows
cv2.destroyAllWindows()
# %%

img = Path('realtime_img')

for images in os.listdir(img):

    if images.endswith('jpg'):

        images = Image.open(os.path.join(img, images)).convert('RGB')
        #print(type(images))
        preProcess_img = transform(images)
        img_tensor = torch.unsqueeze(preProcess_img, 0)
        model.eval()
        out = model(img_tensor)
        
print(out)