# %%
import numpy as np
import torch
import torch as nn
from torchvision.datasets import ImageFolder 
from torchvision import transforms 
import torch.optim as optim 
from torch.utils.data import Subset, SubsetRandomSampler 
import matplotlib.pyplot as plt
import pandas as pd
import os
import pydotplus
import random
import math
from tqdm import tqdm
from collections import defaultdict 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn import metrics

from models import topKResnet18
from resNet_script import compile_dataset, get_dataloaders

# %%
def dataloader_features(dataloader, model, features_idx):
    rows = []
    targets = []
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for (inputs, labels) in tqdm(dataloader):
        inputs = inputs.to(device)
        output_features = model(inputs)
        

        for row in np.squeeze(output_features.to('cpu').detach().numpy()):
            rows.append(row)

        targets += labels.tolist()

        # Here we create a dataframe to store the outputs of the model
    df = pd.DataFrame(rows, columns=features_idx)
    return df, targets

# %%

import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('DATASET_PATH', type=str, help="image data path")
parser.add_argument('OUTPUT_PATH', type=str, help="Path to save outputs")
parser.add_argument('k', type=int, help="number of features for classification")

args = parser.parse_args() 

DATASET_PATH = args.DATASET_PATH
OUTPUT_PATH = args.OUTPUT_PATH
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

# %%

# Gets the top_k features from the training dataset
train_features = model.dim_reduction_layer.df_top_k
train_targets = model.targets

# Gets the top_k feautre's index values
features_idx = model.dim_reduction_layer.top_k_features_idx

# %%
# Extracts the top k features from the test dataset via the dataloader
test_features, test_targets = dataloader_features(test_loader, lambda x: model(x, classify=False), features_idx)

# Splitting the dataset into train and test sets
# features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3, random_state = 100)

# Creating the classifier object

# %%
decision_tree = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100)

# train the decision tree
decision_tree.fit(train_features, train_targets)

# make a prediction using the decision tree
class_prediction = decision_tree.predict(test_features)

# creates a confusion matrix using the decision tree's predictions
class_names = datasets[-1].dataset.classes
result_df = pd.DataFrame(metrics.confusion_matrix(test_targets, class_prediction), class_names, columns=class_names)

# %%
result_df.to_csv(os.path.join(OUTPUT_PATH, 'confusionMatrix_decisionTree.csv'))

# creates png image of the decision tree and stores in output dir 
dot_data = export_graphviz(decision_tree, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png(os.path.join(OUTPUT_PATH, 'decision_tree.png'))

