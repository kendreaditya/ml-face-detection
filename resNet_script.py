import matplotlib.pyplot as plt
import torch 
import torch as nn
from torchvision.datasets import ImageFolder 
from torchvision import transforms 
import torch.optim as optim 
from torch.utils.data import Subset, SubsetRandomSampler 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import math
import random 
from collections import defaultdict 
import copy 
import os 
import argparse
from PIL import Image, ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True 

from models import topKResnet18

# Creates transforms which are used for data augmentation and preprocessing
def get_dataset_transforms():
    np.random.seed(0)
    torch.manual_seed(0)

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

    return {
        'train': transforms.Compose([*data_augmentation, *preprocess]),
        'val': transforms.Compose([transforms.Resize(size=(256, 256)), *preprocess]),
        'test': transforms.Compose([transforms.Resize(size=(256, 256)), *preprocess])
    }

# Creates a Dataset Object that takes the image paths (via an path to the folder) and preprocesses them
def compile_dataset(path):
    dataset = ImageFolder(path)

    dataset_transforms = get_dataset_transforms()
    class_split_freq, [train_dataset, val_dataset, test_dataset] = stratified_split(dataset, [0.7, 0.15, 0.15], dataset_transforms=list(dataset_transforms.values()))

    return train_dataset, val_dataset, test_dataset

# Creates a Dataloader Object that takes the Dataset Object and creates a batch of images
def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=64, num_workers=2):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# Displays a batch of images
def show_batch(batch):
    # Convert the batch to a NumPy array
    batch = batch.numpy()
    
    # Transpose the array so that the channels come last
    batch = np.transpose(batch, (0, 2, 3, 1))
    
    # Create a figure with a grid of images
    fig, ax = plt.subplots(nrows=batch.shape[0], ncols=1)
    
    # Loop over the images in the batch and display them
    for i in range(batch.shape[0]):
        ax[i].imshow(batch[i])
        ax[i].axis('off')
    
    plt.show()

# Creates a stratified split of the dataset
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
                return model

        print(f"Epoch {epoch+1} - Train loss: {epoch_loss:.4f}, Val loss: {val_loss:.4f}")
    return model
    

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

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument('DATASET_PATH', type=str, help="image data path")
    parser.add_argument('CHECKPOINT_PATH', type=str, help="Path to save the model checkpoints")
    parser.add_argument('k', type=int, help="number of features for classification")

    args = parser.parse_args() 

    DATASET_PATH = args.DATASET_PATH
    k = args.k
    
    if os.path.exists(DATASET_PATH):
        print("Path Exists") 
        
    else: 
        print("Path does not exist" )
        exit()

    # Create the dataset and dataloaders
    datasets = compile_dataset(DATASET_PATH)
    train_loader, val_loader, test_loader = get_dataloaders(*datasets)

    # Downloads and initializes the ResNet model from the PyTorch database
    model = topKResnet18(train_loader, k)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train(model,
        train_loader,
        val_loader,
        torch.nn.CrossEntropyLoss(),
        optim.Adam(model.parameters(), lr=0.001),
        device,
        os.path.join(args.CHECKPOINT_PATH, "top-k-model.pt"))
