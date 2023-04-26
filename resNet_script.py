# %%
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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# metrics related
import torchmetrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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
        transforms.CenterCrop(224),  # Crops the image to a 224 x 224 image
        transforms.ToTensor(),
        # Converts the Image Object to a Tensor object (which is how PyTorch keeps track of its gradients which is used for Backpropagation)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the images
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
    class_split_freq, [train_dataset, val_dataset, test_dataset] = stratified_split(dataset, [0.7, 0.15, 0.15],
                                                                                    dataset_transforms=list(
                                                                                        dataset_transforms.values()))

    return train_dataset, val_dataset, test_dataset


# Creates a Dataloader Object that takes the Dataset Object and creates a batch of images
def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=64, num_workers=2):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
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
        splits[i].dataset.classes = dataset.classes

    return class_split_freq, splits


def train_with_accuracy_metrics(model, dataloader, val_dataloader, test_dataloader, criterion, optimizer, device, save_path, dataset_path, early_stop_epochs=5):
    def get_class_names(dataset_path):
        class_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        return class_names

    class_names = get_class_names(dataset_path)
    num_classes = len(class_names)
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Initialize columns for the accuracy DataFrame
    columns = ['epoch'] + class_names + ['total_accuracy']
    accuracy_df = pd.DataFrame(columns=columns)

    for epoch in tqdm(range(20)):  # maximum 20 epochs
        running_loss = 0.0
        model.train()  # set model to train mode
        model = model.to(device)
        accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='none').to(device)

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Update accuracy
            accuracy.update(outputs, labels)

        epoch_loss = running_loss / len(dataloader.dataset)
        per_class_accuracies = accuracy.compute().tolist()  # Get per-class accuracies
        total_accuracy = sum(per_class_accuracies) / num_classes  # Calculate total accuracy

        # Append accuracies to the DataFrame
        accuracy_data = {'epoch': epoch + 1, **{class_names[i]: per_class_accuracies[i] for i in range(num_classes)},
                         'total_accuracy': total_accuracy}
        accuracy_df = accuracy_df.append(accuracy_data, ignore_index=True)

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Total Accuracy: {total_accuracy:.4f}, Per-Class Accuracies: {per_class_accuracies}')

        # Validation
        val_loss = 0.0
        model.eval()  # set model to evaluation mode

        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_dataloader.dataset)
        print(f'Validation Loss: {val_loss:.4f}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f'Best model saved at {save_path}')
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_epochs:
                print('Early stopping triggered')
                break

    # Training loop finished, now evaluate on test set
    model.load_state_dict(torch.load(save_path))  # Load the best model
    model.eval()  # set model to evaluation mode
    test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='none').to(device)

    true_labels = []
    predicted_labels = []

    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        # Update test accuracy
        test_accuracy.update(outputs, labels)

        # Store true labels and predicted labels for confusion matrix
        _, preds = torch.max(outputs, 1)
        true_labels.extend(labels.tolist())
        predicted_labels.extend(preds.tolist())

    per_class_accuracies = test_accuracy.compute().tolist()  # Get per-class accuracies
    total_accuracy = sum(per_class_accuracies) / num_classes  # Calculate total accuracy

    # Append test accuracies to the DataFrame
    accuracy_data = {'epoch': 'test', **{class_names[i]: per_class_accuracies[i] for i in range(num_classes)},
                     'total_accuracy': total_accuracy}
    accuracy_df = accuracy_df.append(accuracy_data, ignore_index=True)

    print(f'Test Total Accuracy: {total_accuracy:.4f}, Test Per-Class Accuracies: {per_class_accuracies}')

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)

    # Save confusion matrix as a heatmap image
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_df, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(f'{save_path[:-3]}_normalized_confusion_matrix.png')

    return model, accuracy_df

def create_animated_accuracy_plot(accuracy_df, save_path):
    fig, ax = plt.subplots()

    def update(epoch):
        ax.clear()
        for class_name in accuracy_df.columns[1:-1]:
            class_accuracy = accuracy_df[class_name].iloc[:epoch + 1]
            ax.plot(class_accuracy, label=class_name)

        ax.set_xlim(0, len(accuracy_df) - 2)  # Exclude the test epoch from the x-axis
        ax.set_ylim(0, 1)  # Set y-axis limits to be between 0 and 1
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Class Accuracies over Epochs')
        ax.legend(loc='upper left')

    ani = FuncAnimation(fig, update, frames=len(accuracy_df) - 1, interval=500, repeat_delay=2000)
    ani.save(save_path, writer='imagemagick', fps=1)


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
        print("Path does not exist")
        exit()

    # Create the dataset and dataloaders
    datasets = compile_dataset(DATASET_PATH)
    train_loader, val_loader, test_loader = get_dataloaders(*datasets)

    # Downloads and initializes the ResNet model from the PyTorch database
    model = topKResnet18(train_loader, k)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_PATH = './checkpoints'
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    # Call the function with the added test_dataloader parameter
    model, accuracy_df = train_with_accuracy_metrics(
        model,
        train_loader,
        val_loader,
        test_loader,  # Add this line
        torch.nn.CrossEntropyLoss(),
        optim.Adam(model.parameters(), lr=0.001),
        device,
        os.path.join(CHECKPOINT_PATH, 'best_model.pt'),
        DATASET_PATH,
        early_stop_epochs=5
    )
    accuracy_df.to_csv(os.path.join(CHECKPOINT_PATH, 'accuracy_metrics.csv'), index=False)
    create_animated_accuracy_plot(accuracy_df, os.path.join(CHECKPOINT_PATH, 'animated_accuracy_plot.gif'))

# %%
