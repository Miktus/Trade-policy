
# Neural net created for the gravity model prediction for the Trade Policy class at PSE
# Author: Michal Miktus at michal.miktus@gmail.com
# Date: 19.02.2019

# Import libraries

import numpy as np
import pandas as pd
import math
import torch
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import mstats

from matplotlib import pyplot as plt
from plotly.offline import init_notebook_mode, iplot
# from plotly import tools
import plotly.graph_objs as go
import plotly.io as pio


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
description.sort_values(by="cova", axis=1)

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

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)],
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

# This implementation defines the model as a custom Module subclass. Whenever you
# want a model more complex than a simple sequence of existing Modules you will
# need to define your model this way.


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


x = torch.tensor(x_train).float()
y = torch.tensor(y_train).float()

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = int(data_PL.shape[0]), int((data_PL.shape[1] - 1)), 10, 1

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
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

model = Sequential()

model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=50,
    shuffle=True,
    verbose=2)

predictions = model.predict(x_test)

# Second approach to TensorFlow


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=[x_train.shape[1]]),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 1000

history = model.fit(
    x_train, y_train,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 5])

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 20])


plot_history(history)

# Early stopping

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(x_train, y_train, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

# Loss on test set

loss, mae, mse = model.evaluate(x_test, y_test, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} flow".format(mae))

# Make predictions

test_predictions = model.predict(x_test).flatten()

plt.scatter(y_test, test_predictions);


# Error distribution

error = test_predictions - y_test
plt.hist(error, bins=25);

# TO ADD
# L1/L2 regularization, dropout regularization
# He initialization/Xavier initialization
# Learning rate decay: SGD with momentum, rmsprop, adam
# Batch normalization
# Hyperparameter tuning: grid/random grid
