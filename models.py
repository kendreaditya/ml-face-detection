import torch 
import torch as nn
import numpy as np 
import pandas as pd
from tqdm import tqdm

class DimensionalityReduction(torch.nn.Module):
    def __init__(self, df, out_features=64):
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
      mask = np.tril(np.ones(corr_matrix.shape)).astype(bool)

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

class topKResnet18(torch.nn.Module):
    def __init__(self, train_loader, k):
        super(topKResnet18, self).__init__()
        self.n_classes = len(train_loader.dataset.dataset.classes)
        print(self.n_classes)

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        # Here we remove the last layer - fc (fully connected) so we can do PCA on it
        model = nn.nn.Sequential(*list(model.children())[:-1])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(self.device)

        # Here we run the dataset though the model to get the raw features with the shape of (batch_size, 512)
        rows = []
        for (inputs, labels) in tqdm(train_loader):
            inputs = inputs.to(self.device)
            output_features = model(inputs)

            for row in np.squeeze(output_features.to('cpu').detach().numpy()):
                rows.append(row)

        # Here we create a dataframe to store the outputs of the model
        df = pd.DataFrame(rows, columns=[f'{i}' for i in range(512)])

        for param in model.parameters():
            param.requires_grad = False

        self.feature_extraction = torch.nn.Sequential(*list(model.children()), DimensionalityReduction(df, out_features=k))
        self.classification = torch.nn.Linear(in_features=k, out_features=self.n_classes)

        self.feature_extraction.to(self.device)
        self.classification.to(self.device)

    def forward(self, x, classify=True):
        features = self.feature_extraction(x)

        if classify:
            return self.classification(features)

        return features
