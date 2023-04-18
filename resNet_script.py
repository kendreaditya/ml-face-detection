#imports 

import torch 
import torch as nn
from torchvision.datasets import ImageFolder 
from torchvision import transforms 
import torch.optim as optim 

import numpy as np 
import pandas as pd 

from tqdm import tqdm 

import math
import random 
from collections import defaultdict 
from torch.utils.data import Subset, SubsetRandomSampler 
import copy 
import os 
import argparse
from PIL import Image, ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True 

#Data Preprocessing 

preprocess = transforms.Compose([

    # Data Augmentation
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # transforms.RandomRotation(degrees=10),
    # transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2), ratio=(0.75, 1.33)),

    # Data Preprocessing
    transforms.CenterCrop(224), # Crops the image to a 224 x 224 image
    transforms.ToTensor(), # Converts the Image Object to a Tensor object (which is how PyTorch keeps track of its gradients which is used for Backpropagation)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the images
])


#@title Create Dataset & Dataloader
# Create a Dataset Object that takes the image paths (via an path to the folder) and preprocesses them
#Need to define DATASET_PATH, make it user-entered 

#print(f"Enter dataset path")
#DATASET_PATH = input()

#if os.path.exists(DATASET_PATH):
#    print("Path Exists") 


dataset = ImageFolder(DATASET_PATH, preprocess)


def stratified_split(dataset, split_ratios):
    num_splits = len(split_ratios)
    # Compute class frequencies
    class_freq = defaultdict(int)
    for label in dataset.targets:
        class_freq[label] += 1

    # Compute number of samples per class per split
    num_samples_per_class_per_split = {}
    for label, freq in class_freq.items():
        num_samples_per_class_per_split[label] = [math.ceil(freq * split_ratio) for split_ratio in split_ratios]


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

    return class_split_freq, splits


class_split_freq, [train_dataset, val_dataset, test_dataset] = stratified_split(dataset, [0.7, 0.15, 0.15])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

# Downloads and initializes the ResNet model from the PyTorch database
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Here we remove the last layer - fc (fully connected) so we can do PCA on it
model = nn.nn.Sequential(*list(model.children())[:-1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Here we run the dataset though the model to get the raw features with the shape of (batch_size, 512)
rows = []
for (inputs, labels) in tqdm(train_loader):
  inputs = inputs.to(device)
  output_features = model(inputs)

  for row in np.squeeze(output_features.to('cpu').detach().numpy()):
    rows.append(row)

# Here we create a dataframe to store the outputs of the model
df = pd.DataFrame(rows, columns=[f'{i}' for i in range(512)])

df.to_csv("../ml-face-detection/ResNet-features.csv")


# Define custom layer
class DimensionalityReduction(torch.nn.Module):
    def __init__(self, df, in_features=512, out_features=64):
        super(DimensionalityReduction, self).__init__()
        df_top_k = self.find_k_top_corr(df, out_features)

        self.columns = [int(i) for i in list(df_top_k.columns)]

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

#k is a parameter that will need to be user-entered
print("Enter the number of parameters for the model: ") 
k = 64
n_classes = len(dataset.classes)
df = pd.read_csv("../ml-face-detection/ResNet-features.csv")

# Downloads and initializes the ResNet model from the PyTorch database
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

for param in model.parameters():
    param.requires_grad = False

top_k_model = torch.nn.Sequential(*list(model.children())[:-1], DimensionalityReduction(df, in_features=512, out_features=k), torch.nn.Linear(in_features=64, out_features=n_classes))

def train(model, dataloader, val_dataloader, criterion, optimizer, device, save_path, early_stop_epochs=5):
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in tqdm(range(100)):  # maximum 100 epochs
        running_loss = 0.0
        model.train()  # set model to train mode
        model = model.to(device)

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)

        # validation
        val_loss = 0.0
        model.eval()  # set model to eval mode

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_dataloader.dataset)

            # save checkpoint if val loss is better
            if val_loss < best_val_loss:
                print(f"Saving model checkpoint with validation loss of {val_loss:.4f}")
                torch.save(model.state_dict(), save_path)
                best_val_loss = val_loss
                early_stop_counter = 0  # reset early stop counter
            else:
                early_stop_counter += 1

            # stop early if overfitting detected
            if early_stop_counter == early_stop_epochs:
                print(f"Stopping training after {epoch} epochs due to overfitting")
                return

        print(f"Epoch {epoch+1} - Train loss: {epoch_loss:.4f}, Val loss: {val_loss:.4f}")

train(top_k_model, train_loader,
     val_loader, nn.CrossEntropyLoss(),
     optim.SGD(top_k_model.parameters(), lr=0.001, momentum=0.9),
     device,
      "../ml-face-detection/top-k-model.pt")


def calculate_metrics(model, val_loader, class_names):
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Get the device the model is on

    total_samples = 0
    correct_samples = 0
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to device

            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get predicted classes

            # Update overall and per-class metrics
            total_samples += labels.size(0)
            correct_samples += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    accuracy = 100 * correct_samples / total_samples
    class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(len(class_names))]

    print('Accuracy: {:.2f}%'.format(accuracy))
    for i in range(len(class_names)):
        print('Class {}: {:.2f}%'.format(class_names[i], class_accuracy[i]))

    return accuracy, class_accuracy

checkpoint_path = '../ml-face-detection/top-k-model.pt'

# Load the checkpoint file
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Extract the model state dictionary from the checkpoint
# Create a new instance of your model
# Load the state dictionary into the model
top_k_model.load_state_dict(checkpoint)

calculate_metrics(top_k_model, val_loader, os.listdir(DATASET_PATH))
device
calculate_metrics(top_k_model, test_loader, os.listdir(DATASET_PATH))

#Argparse section 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

