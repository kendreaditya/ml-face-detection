# imports needed
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

if __name__ == "__main__":

    # gets data as pandas data frame from script args
    import argparse

    parser = argparse.ArgumentParser() 
    parser.add_argument('DATASET_PATH', type=str, help="image data path")
    parser.add_argument('OUTPUT_PATH', type=str, help="Path to save metrics")
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

    # get features from resnet model and convert to df
    features = model.predict(datasets)
    features_df = pd.DataFrame(features.reshape(features.shape[0], -1))

    # split the data into training and testing sets

    # Separating the target variable
    features = features_df.values[:, 1:5]
    target = features_df.values[:, 0]

    # Splitting the dataset into train and test sets
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3, random_state = 100)

    # Creating the classifier object
    decision_tree = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100)

    # train the decision tree
    decision_tree.fit(features_train, target_train)

    # make a prediction using the decision tree
    class_prediction = decision_tree.predict(features_test)

    # creates a confusion matrix using the decision tree's predictions
    result_df = metrics.confusion_matrix(target_test, class_prediction)

    result_df.to_csv(os.path.join(args.output_dir, 'confusionMatrix_decisionTree.csv'))

    # creates png image of the decision tree and stores in output dir 
    dot_data = export_graphviz(decision_tree, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(os.path.join(args.output_dir, 'decision_tree.png'))
    
