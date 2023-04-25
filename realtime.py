# %%
import cv2
import numpy as np
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
from torchvision import transforms
from PIL import Image

# %%
from models import topKResnet18
model = topKResnet18(None, topK=64, columns=[f'{i}' for i in range(512)])

#Create video capture object

# %%
vid = cv2.VideoCapture(0)
arr = []
preProcess_img = []
transform = transforms.Compose([
    # Data Preprocessing
    transforms.CenterCrop(224), # Crops the image to a 224 x 224 image 
    transforms.ToTensor(), 
    # Converts the Image Object to a Tensor object (which is how PyTorch keeps track of its gradients which is used for Backpropagation
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the images
])


cv2.namedWindow("Window")

# %%
while(vid.isOpened()):

    #capture video frame by frame
    ret, frame = vid.read()
    im = Image.fromarray(frame)
    preprocessed = transform(im)
    pred = model(preprocessed.unsqueeze(0))
    print(pred.shape)
    break

    cv2.imshow('Window', frame)

    #quit the script using the q key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(len(arr))
        arr
        for image in arr:
            preProcessed = transform(image)
            print(preProcessed.shape)
            # preProcess_img = np.append(preProcess_img, preProcessed)
        #Feed data into model
        preProcess_img = torch.stack(preProcess_img)
        model(preProcess_img).shape
        break


#Release video capture object
vid.release()

#destroy all windows
cv2.destroyAllWindows()
# %%
