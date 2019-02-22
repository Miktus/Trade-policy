
# Neural net created for the gravity model prediction for the Trade Policy class at PSE
# Author: Michal Miktus at michal.miktus@gmail.com
# Date: 19.02.2019

# Import libraries

import plotly.io as pio
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from matplotlib import pyplot as plt
from scipy.stats import mstats
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
import pandas as pd
import math
import torch
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import talos as ta
# from talos.model.layers import hidden_layers
# from talos.model.normalizers import lr_normalizer


# from plotly import tools


# Set seed

random_state = 123
np.random.seed(random_state)
tf.set_random_seed(random_state)
torch.manual_seed(random_state)

# Supress scientific notation for pandas

pd.options.display.float_format = '{:.5f}'.format

# Templates for graphs

pio.templates.default = 'plotly_dark+presentation'
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
init_notebook_mode(connected=True)


# Path specifiation

path = "/Users/miktus/Documents/PSE/Trade policy/Model/"

# Import data

data = pd.read_stata(path + "/Data/col_regfile09.dta")

# Data exploration

data

data.info()

# Dropping the duplicates from the dataset

data = data.drop_duplicates(keep='first')

# Handling missing data

data.isnull().sum()

data.dropna(thresh=data.shape[0] * 0.6, how='all', axis=1, inplace=True)

data.dropna(axis=0, inplace=True)
# data.fillna(data.mean(), inplace=True) # Or replace by the column mean

# Desribe data

description = data.describe(include='all')
coef_variation = description.loc["std"] / description.loc["mean"]
description.loc["cova"] = coef_variation
(description.sort_values(by="cova", axis=1)).T

# Number of unique entries

print(data.nunique())

# Names of binary data (unstandarized)

binary = []
for columns in data:
    if (data.loc[:, columns].min() == 0) & (data.loc[:, columns].max() == 1):
        binary.append(columns)

for columns in data.loc[:, binary]:
    print(data.loc[:, binary][columns].unique())

# Remove iso_2o, iso_2d and family

data.drop(columns=['iso2_d', 'iso2_o', 'family'], inplace=True)

# Numeric variables

data_numeric = data._get_numeric_data()
data_numeric.drop(columns="year", inplace=True)
data_numeric.drop(columns=binary, inplace=True)

# Numerical data distribution

data_numeric.hist(figsize=(10, 10), bins=50, xlabelsize=8, ylabelsize=8)

# Normalization

minmax_normalized_df = pd.DataFrame(MinMaxScaler().fit_transform(data_numeric),
                                    columns=data_numeric.columns, index=data_numeric.index)

standardized_df = pd.DataFrame(StandardScaler().fit_transform(data_numeric), columns=data_numeric.columns,
                               index=data_numeric.index)

ecdf_normalized_df = data_numeric.apply(
    lambda c: pd.Series(ECDF(c)(c), index=c.index))

# Replace data by its standardized values

data[list(ecdf_normalized_df.columns.values)] = ecdf_normalized_df


# Visualisations

# Flows
print(data['flow'].describe())

flows_winsorized = mstats.winsorize(data['flow'], limits=[0.05, 0.05])
layout = go.Layout(
    title="Basic histogram of flows (winsorized)")

data_hist = [go.Histogram(x=flows_winsorized)]
fig = go.Figure(data=data_hist, layout=layout)

iplot(fig, filename='Basic histogram of flows')


# Corr

corr = ecdf_normalized_df.corr()

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)],
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)

# Coef of variation

layout = go.Layout(
    title="Coefficient of variation")

data_cova = [go.Histogram(x=description.loc["cova"])]
fig = go.Figure(data=data_cova, layout=layout)
iplot(fig,
      filename="Coefficient of variation")

high_cova = description.loc["cova"].where(
    lambda x: x > 0.30).dropna().sort_values(ascending=False)
high_cova

# Select only POL as iso_o

data_PL = data.query("iso_o == 'POL'")

# One hot encoding
data_PL = pd.get_dummies(
    data_PL, columns=["iso_o", "iso_d"], prefix=["iso_o", "iso_d"])

# Splitting the data

train_size = 0.9

train_cnt = math.floor(data_PL.shape[0] * train_size)
x_train = data_PL.drop('flow', axis=1).iloc[0:train_cnt].values
y_train = data_PL.loc[:, 'flow'].iloc[0:train_cnt].values
x_test = data_PL.drop('flow', axis=1).iloc[train_cnt:].values
y_test = data_PL.loc[:, 'flow'].iloc[train_cnt:].values

# Build NN class in PyTorch

# A fully-connected ReLU network with one hidden layer, trained to predict y from x
# by minimizing squared Euclidean distance.


class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H_in, H_out, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H_in)
        self.linear2 = torch.nn.Linear(H_in, H_out)
        self.linear3 = torch.nn.Linear(H_out, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu_1 = self.linear1(x).clamp(min=0)
        h_relu_2 = self.linear2(h_relu_1).clamp(min=0)
        y_pred = self.linear3(h_relu_2)
        return y_pred


x = torch.tensor(x_train).float()
y = torch.tensor(y_train).float()

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H_in, H_out, D_out = int(data_PL.shape[0]), int((data_PL.shape[1] - 1)), 50, 50, 1

# Construct our model by instantiating the class defined above
model = ThreeLayerNet(D_in, H_in, H_out, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the three
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(5):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Build NN class in TensorFlow


def build_model(x_train, y_train, x_val, y_val, params):

    model = keras.Sequential()
    model.add(keras.layers.Dense(params['first_neuron'], activation=params['activation'])
                                 # input_shape=[x_train.shape[1]])
                                 # use_bias=True,
                                 # kernel_initializer='glorot_uniform',
                                 # bias_initializer='zeros',
                                 # kernel_regularizer=keras.regularizers.l1_l2(l1=params['l1'], l2=params['l2']),
                                 # bias_regularizer=None))

    # If we want to also test for number of layers and shapes, that's possible
    # hidden_layers(model, params, 1)

    # Then we finish again with completely standard Keras way
    model.add(keras.layers.Dense(1, activation=params['activation'],
                                 kernel_initializer='glorot_uniform'))

    model.compile(optimizer=params['optimizer'],
                  loss=params['losses'])
    # metrics=['mae', 'mse'])

    history=model.fit(x_train, y_train,
                        validation_data=[x_val, y_val],
                        # batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0)

    # finally we have to make sure that history object and model are returned
    return history, model


# then we can go ahead and set the parameter space
p={'lr': (0.5, 5, 10),
     # 'l1': [0.1, 0.2],
     # 'l2': [0.1, 0.2],
     'first_neuron': [4, 8, 16, 32, 64],
     # 'hidden_layers': [0, 1, 2],
     # 'batch_size': (2, 3, 4),
     'epochs': [150],
     # 'dropout': (0, 0.5, 5),
     'weight_regulizer': [None],
     'emb_output_dims': [None],
     'optimizer': ['Adam', 'Nadam', 'SGD'],
     'losses': ['MSE', 'MAE'],
     'activation': ['tanh', 'relu', 'sigmoid']}

# and run the experiment
t=ta.Scan(x=x_train,
            y=y_train,
            model=build_model,
            grid_downsample=0.01,
            params=p,
            experiment_no='2')
