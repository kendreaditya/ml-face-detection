# imports needed
import pandas as pd
import os
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn import metrics

if __name__ == "__main__":

    # gets data as pandas data frame from script args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = pd.DataFrame, help = "pandas dataframe containing data")
    parser.add_argument("--output_dir", type=str, default="./DecTreeMetrics", help="path to output directory")

    args = parser.parse_args()

    data = args.data

    # split the data into training and testing sets

    # Separating the target variable
    features = data.values[:, 1:5]
    target = data.values[:, 0]

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
