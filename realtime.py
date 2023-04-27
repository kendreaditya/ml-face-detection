# %%
import cv2
import numpy as np
import torch
import os 
from pathlib import Path
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# %%
DATASET_PATH = './images/'
df = pd.read_csv("./top-64-ResNet-features.csv")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
checkpoint_path = './top-k-model.pt'
# %%
data_augmentation = [
    # Data Augmentation
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2), ratio=(0.75, 1.33))
]

preprocess = [
    
    # Data Preprocessing
    transforms.CenterCrop(224), # Crops the image to a 224 x 224 image 
    transforms.ToTensor(), # Converts the Image Object to a Tensor object (which is how PyTorch keeps track of its gradients which is used for Backpropagation)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the images
]

dataset_transforms = {
    'train': transforms.Compose([*data_augmentation, *preprocess]),
    'val': transforms.Compose([transforms.Resize(size=(256, 256)), *preprocess]),
    'test': transforms.Compose([transforms.Resize(size=(256, 256)), *preprocess])
}

dataset = ImageFolder(DATASET_PATH)

import math
import random
from collections import defaultdict
from torch.utils.data import Subset, SubsetRandomSampler

def stratified_split(dataset, split_ratios, dataset_transforms=None):
    num_splits = len(split_ratios)
    # Compute class frequencies
    class_freq = defaultdict(int)
    for label in dataset.targets:
        class_freq[label] += 1

    # Compute number of samples per class per split
    num_samples_per_class_per_split = {}
    for label, freq in class_freq.items():
        num_samples_per_class_per_split[label] = [math.ceil(freq * split_ratio) for split_ratio in split_ratios]
    
    import copy
    class_split_freq = copy.deepcopy(num_samples_per_class_per_split)
    
    # Create empty lists for each split
    split_indices = []
    for i in range(num_splits):
        split_indices.append([])

    # Assign indices to each split based on class frequency
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for idx in indices:
        label = dataset.targets[idx]
        for i in range(num_splits):
            if num_samples_per_class_per_split[label][i] > 0:
                split_indices[i].append(idx)
                num_samples_per_class_per_split[label][i] -= 1

    # Create Subset objects for each split
    splits = []
    for i in range(num_splits):
        split_sampler = SubsetRandomSampler(split_indices[i])
        split_dataset = Subset(dataset, split_indices[i])
        splits.append(split_dataset)
    
    for i in range(len(splits)):
      splits[i] = copy.deepcopy(splits[i])
      splits[i].dataset.transform = dataset_transforms[i]
    return class_split_freq, splits


class_split_freq, [train_dataset, val_dataset, test_dataset] = stratified_split(dataset, [0.7, 0.15, 0.15], dataset_transforms=list(dataset_transforms.values()))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=2)

# Downloads and initializes the ResNet model from the PyTorch database
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""#Pick TOP 64 correlated features from ResNet Feature Extraction"""

# Define custom layer
class DimensionalityReduction(nn.Module):
    def __init__(self, df, in_features=512, out_features=64, columns = None):
        super(DimensionalityReduction, self).__init__()
        if columns == None:
          df_top_k = self.find_k_top_corr(df, out_features)
          self.columns = [int(i) for i in list(df_top_k.columns)]        
        else:
          self.columns = columns 
    
    def find_k_top_corr(self, df, number):
      """
        Given a pandas dataframe, and number of columns you want, it returns df with top n correlated columns
    
        Parameters:
        -----------
        df : pandas dataframe
        n  : the number of columns you want as the most correlated columns(features)
    
        Returns:
        --------
        reduced_df : DataFrame
            pandas dataframe with top n most correlated features
        """
      # Calculate the correlation matrix(abs for absolute values, focus on correlation itself)
      corr_matrix = df.corr().abs()
    
      # lets create a mask with size of our corr matrix that filled with ones,  but it will only fill lower trinagle. then convert them as boolean.
      mask = np.tril(np.ones(corr_matrix.shape)).astype(np.bool)
    
      # with using df.mask, we fill the lower triangular matrix to NaN(includes diagonal with 1.0s)
      masked_corr_matrix = corr_matrix.mask(mask)
    
      # using df.unstack() we turn them into a new level of column labels whose inner-most level consists of the pivoted index labels.
      # In summary, we can get new table that each index of feature that has all the correlations of each every other features
      # https://www.w3resource.com/pandas/dataframe/dataframe-unstack.php
      correlations_sorted_table = masked_corr_matrix.unstack(level=1).sort_values(ascending=False)
    
      # now we got every correlation of the matrix in a long line of index, we search top 64 of them without duplicates.
      top_features = []
      for pair in correlations_sorted_table.index:
          # add first pair if length did not reach
          if pair[0] not in top_features and len(top_features) < number:
              top_features.append(pair[0])
          # add second pair if length did not reach
          if pair[1] not in top_features and len(top_features) < number:
              top_features.append(pair[1])
          # break if length reached
          if len(top_features) >= number:
              break
    
      # only pick the columns th"/data" at has top 64 from previous dataframe to our new dataframe
      reduced_df = df[top_features]
    
      return reduced_df

    def forward(self, x):
        # Select only the columns specified in self.columns
        reduced_x = x[:, self.columns].squeeze()

        return reduced_x

k = 64
n_classes = len(dataset.classes)
# %%
for param in model.parameters():
    param.requires_grad = False

top_k_model = torch.nn.Sequential(*list(model.children())[:-1], DimensionalityReduction(None, in_features=512, out_features=k, columns=list(map(int, df.columns[1:]))), nn.Linear(in_features=64, out_features=n_classes))

checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# %%
top_k_model.load_state_dict(checkpoint)

import torch
import matplotlib.pyplot as plt
from PIL import Image

def import_image(path, transforms=dataset_transforms['test']):
  image = Image.open(path)
  image = transforms(image)
  return image

def predict(image, model, classes):
    """
    Predict the class of an input image using the provided model.
    
    Args:
        image_path (str): Path to the input image file.
        model: The trained model to use for prediction.
        classes (list): A list of the class names.
        
    Returns:
        None
    """
    
    # Make a prediction using the model
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
    idx = torch.argmax(output)
    predicted_class = classes[idx]
    
    # Display the input image and the predicted class
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'Predicted class: {predicted_class}')
    plt.show()

classes = train_loader.dataset.dataset.classes
# %%
vid = cv2.VideoCapture(0)
cv2.namedWindow("Window")
# %%
count = 0
while(vid.isOpened()):
    #capture video frame by frame
    ret, frame = vid.read()
    #print(type(frame))
    # cv2.imwrite(f'realtime_img\\frame%d.jpg' % count, frame)
    im = Image.fromarray(frame)
    preprocessed = dataset_transforms['test'](im)
    pred = top_k_model(preprocessed.unsqueeze(0))
    class_pred = classes[torch.argmax(pred)]
    cv2.imshow(class_pred, frame)
    
    #quit the script using the q key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


#Release video capture object
vid.release()
# %

#destroy all windows
cv2.destroyAllWindows()
# %%