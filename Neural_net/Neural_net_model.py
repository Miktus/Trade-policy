# Authors: Michal Miktus at michal.miktus@gmail.com
#          Mateusz Szmidt at mateuszszmidt95@gmail.com
# Neural net created for the gravity model prediction for the Trade Policy class at PSE
# Date: 08.04.2019

# Import libraries

import plotly.io as pio
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
from matplotlib import pyplot as plt
from scipy.stats import mstats
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import mode

import os
import numpy as np
import pandas as pd
#import torch
import seaborn as sns
import keras
import tensorflow as tf
import talos as ta
from keras.optimizers import Adam, Nadam, SGD
from keras.activations import relu, elu, sigmoid, tanh
from keras.losses import mse
from talos.model.normalizers import lr_normalizer
from talos.model.layers import hidden_layers
from talos.model.early_stopper import early_stopper
from talos import Evaluate
%matplotlib inline

# from plotly import tools

pd.options.display.float_format = '{:.2f}'.format

# Set seed
random_state = 123
np.random.seed(random_state)
tf.set_random_seed(random_state)
# torch.manual_seed(random_state)

# Supress scientific notation for pandas

pd.options.display.float_format = '{:.5f}'.format

# Templates for graphs

# pio.templates.default = 'plotly_dark+presentation'
sns.set(style="ticks", context="talk")
plt.style.use("seaborn")
init_notebook_mode(connected=True)

# Path specifiation
path = "/Users/miktus/Documents/PSE/Trade policy/Model/"
# path = "C:/Repo/Trade/Trade-policy/"

# Import data

data = pd.read_csv(path + "/Data/final_data_trade.csv")

# Data exploration only for Poland

data = data.loc[data['rt3ISO'] == "POL"]
data = data.loc[data['yr'] > 1993]
min(data['yr'])
data.shape
data.columns

# Number of trade partners

data["pt3ISO"].unique().shape

data.info()

# Dropping the duplicates from the dataset

data = data.drop_duplicates(keep='first')

# Handling missing data

data.isnull().sum()

data.dropna(thresh=data.shape[0] * 0.7, how='all', axis=1, inplace=True)

data.dropna(axis=0, inplace=True)
# data.fillna(data.mean(), inplace=True) # Or replace by the column mean

# Desribe data

description = data.describe(include='all')
description.loc['count'] = pd.to_numeric(description.loc['count'])
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

data.drop(columns=['iso2_d', 'iso2_o'], inplace=True)

# Filtering the data for Poland only

data = data.query("rt3ISO == 'POL'")

# Summary statistics table
values = pd.DataFrame(data.nunique(0), columns=["count"])
values["column"] = values.index
keep = values["column"].loc[values["count"] > 1]
data = data[data.columns.intersection(list(keep.append(pd.Series(["rt3ISO"]))))]
values = pd.DataFrame(data.nunique(0), columns=["count"])
values["column"] = values.index
discrete = values["column"].loc[values["count"] <= 10]
continues = values["column"].loc[values["count"] > 10]
continues = pd.DataFrame(data[data.columns.intersection(list(continues.drop(["pt3ISO", "yr"])))].describe(include='all').transpose())
continues["Description"] = ["Total value of trade between reporting and partner countries",
                            "Weighted bilateral distance between reporting and partner countries in kilometer (population weighted)",
                            "Population of reporting country, total in million",
                            "Population of partner country, total in million",
                            "GDP of reporting country (current US$)",
                            "GDP of partner country (current US$)",
                            "GDP per capita of reporting country (current US$)",
                            "GDP per capita of partner country (current US$)",
                            "Area of partner country in sq. kilometers",
                            "Time difference between reporting and partner countries, in number of hours. For countries which stretch over more than  one  time  zone,  the  respective  time  zone  is generated via the mean of all its time zones (for instance: Russia, Canada, USA)",
                            "Religious proximity (Disdier and Mayer, 2007) is an index calculated by adding the productsof the shares of Catholics, Protestants and Muslims in the exporting and importing countries. It is bounded between 0 and 1, and is maximum if the country pair has a religion which (1) comprises a vast majority of the population, and (2) is the same in both countries. Source of religion shares: LaPorta, Lopez-de-Silanes, Shleiferand Vishny(1999), completed with the CIA world factbook"]

# Final table for continues variables
pd.DataFrame(continues[continues.columns.drop(["count"])]).style.format({'total_amt_usd_pct_diff': "{:.2%}"})

discrete = pd.DataFrame(data[data.columns.intersection(list(discrete.append(pd.Series(["yr", "pt3ISO"]))))]).apply(lambda r: pd.Series({'Count': r.nunique(), 'Most Common Value': str(mode(r)[0]).replace("[", "").replace("]", "").replace(".", "")})).transpose()
discrete["Description"] = ["Year",
                           "Standard ISO code for reporting country (three letters)",
                           "Standard ISO code for partner country (three letters)",
                           "Dummy for contiguity",
                           "Dummy if parter country is current or former hegemon of origin",
                           "Dummy for reporting and partner countries colonial relationship post 1945",
                           "Dummy for reporting and partner countries ever in colonial relationship",
                           "Dummy for reporting and partner countries ever in sibling relationship, i.e. two colonies of the same empire",
                           "Dummy if reporting and partner countries share common legal origins before transition",
                           "Dummy if reporting and partner countries share common legal origins after transition",
                           "Dummy if common legal origin changed since transition",
                           "Legal system of partner country before transition. This variable takes the values: “fr” for French, “ge” for German, “sc” for Scandinavian, “so” for Socialist and “uk” for British legal origin.",
                           "Legal system of partner country after transition. This variable takes the values: “fr” for French, “ge” for German, “sc” for Scandinavian, “so” for Socialist and “uk” for British legal origin.",
                           "Dummy if partner country is GATT/WTO member",
                           "Dummy for Regional Trade Agreement",
                           "Dummy for ACP country exporting to EC/EU member",
                           "Dummy if origin is donator in Generalized System of Preferences (GSP)",
                           "Report changes in Rose’s data on <gsp_o_d>. No gsp recorded in Rose; Data directly from Rose; Changes in data from Rose; Assumption that gsp continues after 1999",
                           "Dummy if reporting country a member of the European Union",
                           "Dummy if partner country a member of the European Union"]

discrete.index

# Numeric variables

data_numeric = data._get_numeric_data()
data_numeric.drop(columns="yr", inplace=True)
data_numeric.drop(columns=binary, inplace=True)

# Selected Visualisations

# Histogram of flows over the history

hist_all = sns.distplot(np.log(data["Trade_value_total"] + 1), axlabel="Logarithm of flows", color="blue")


hist_all.figure.savefig('Histogram of flows over the history.png', bbox_inches="tight")

# Histograms for chosen years
years = (1994, 2000, 2009, 2015)
for i in years:
    plt.figure(i)
    hist_temp = sns.distplot(np.log(data["Trade_value_total"].loc[data['yr'] == i] + 1), axlabel="Logarithm of flows in year " + str(i), color="blue")
    hist_temp.figure.savefig('Histogram of flows for ' + str(i) + '.png', bbox_inches="tight")


# Pairplot for distance, Trade_value_total and gdp - choose data and if needed logarithms of values
data_pairplot = data_numeric[["Trade_value_total", "distw", "gdp_d"]]
pairplot = sns.pairplot(data_pairplot, vars=["Trade_value_total", "distw", "gdp_d"], kind="scatter", markers=".",  diag_kind="kde",
                        plot_kws=dict(s=50, edgecolor="blue", linewidth=1),  diag_kws=dict(shade=True,  color="blue"))
pairplot.savefig('Pairplots.png', bbox_inches="tight")

# Save copy of nostandardized dataset
data_nonstandardized = data

data_PL_nonstd = data_nonstandardized.query("rt3ISO == 'POL'")
# data_PL.to_csv("data_PL2.csv")
data_PL_nonstd["year"] = data_PL_nonstd["yr"]
data_PL_nonstd.drop('rt3ISO', axis=1, inplace=True)

# One hot encoding
data_PL_nonstd = pd.get_dummies(
    data_PL_nonstd, columns=["year", "pt3ISO", "legold_d", "legnew_d", "flaggsp_o_d"],
    prefix=["yr", "pt3ISO", "legold_d", "legnew_d", "flaggsp_o_d"])

data_PL_nonstd.to_csv("data_PL.csv")

# Normalization
minmax_normalized_df = pd.DataFrame(MinMaxScaler().fit_transform(data_numeric),
                                    columns=data_numeric.columns, index=data_numeric.index)

standardized_df = pd.DataFrame(StandardScaler().fit_transform(data_numeric), columns=data_numeric.columns,
                               index=data_numeric.index)

ecdf_normalized_df = data_numeric.apply(
    lambda c: pd.Series(ECDF(c)(c), index=c.index))

# Continue with standardized data for neural network
data[list(standardized_df.columns.values)] = standardized_df


# Heatmap
corr = standardized_df.corr()
heat = sns.heatmap(corr[(corr >= 0.3) | (corr <= -0.3)],
                   cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.05,
                   annot=True, annot_kws={"size": 5}, square=True)

heat.figure.savefig('Heatmap.png', bbox_inches="tight")

# Visualise flows - you can choose two parameters
scope = 'world'
# or 'europe'
flow_treshold = 0.92


flows = data[['yr', 'rt3ISO', 'pt3ISO', 'Trade_value_total']]
data_loc = pd.read_csv(path + "/Data/CountryLatLong.csv")
data_loc.drop(columns=['Country'], inplace=True)
data_loc.columns = ["CODE", "rt_Lat", "rt_Long"]

flows = pd.merge(flows, data_loc, left_on="rt3ISO", right_on="CODE").drop('CODE', axis=1)
data_loc.columns = ["CODE", "pt_Lat", "pt_Long"]
flows = pd.merge(flows, data_loc, left_on="pt3ISO", right_on="CODE").drop('CODE', axis=1)

flow_directions = []
for i in range(len(flows)):
    if (flows['Trade_value_total'][i] > flow_treshold):
        flow_directions.append(
            dict(
                type='scattergeo',
                locationmode='ISO-3',
                lon=[flows['rt_Long'][i], flows['pt_Long'][i]],
                lat=[flows['rt_Lat'][i], flows['pt_Lat'][i]],
                text=flows['pt3ISO'][i],
                mode='lines',
                line=dict(
                    width=flows['Trade_value_total'][i] * 10,
                    color='blue',
                ),
                #opacity = 0,5 * (float(flows['yr'][i])/1994)
                opacity=np.power(float(flows['yr'][i]) - float(flows['yr'].min()), 2)/10/float(np.power(float(flows['yr'].max() - float(flows['yr'].min())), 2)),
            )
        )


layout = dict(
        title='Trade flows between Poland and its trading partners.',
        showlegend=False,
        geo=dict(
            scope=scope,
            projection=dict(type='robinson'),
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
        )
    )

fig = dict(data=flow_directions, layout=layout)
plot(fig, filename='Flows map')


# Select only POL as rt3ISO
data_PL = data.query("rt3ISO == 'POL'")
# data_PL.to_csv("data_PL2.csv")
data_PL["year"] = data_PL["yr"]
data_PL.drop('rt3ISO', axis=1, inplace=True)

# One hot encoding
data_PL = pd.get_dummies(
    data_PL, columns=["year", "pt3ISO", "legold_d", "legnew_d", "flaggsp_o_d"],
    prefix=["yr", "pt3ISO", "legold_d", "legnew_d", "flaggsp_o_d"])

splitting_yr = 2010

x_train = data_PL.drop('yr', axis=1).drop('Trade_value_total', axis=1).loc[data_PL['yr'] <= splitting_yr].values
y_train = data_PL.loc[:, 'Trade_value_total'].loc[data_PL['yr'] <= splitting_yr].values
x_test = data_PL.drop('yr', axis=1).drop('Trade_value_total', axis=1).loc[data_PL['yr'] > splitting_yr].values
y_test = data_PL.loc[:, 'Trade_value_total'].loc[data_PL['yr'] > splitting_yr].values
sns.distplot(y_train, axlabel="Logarithm of flows", color="blue")


# Build NN class in Keras
def build_model(x_train, y_train, x_val, y_val, params):

    model = keras.Sequential()
    model.add(keras.layers.Dense(10, activation=params['activation'],
                                 input_dim=x_train.shape[1],
                                 use_bias=True,
                                 kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=keras.regularizers.l1_l2(l1=params['l1'], l2=params['l2']),
                                 bias_regularizer=None))

    model.add(keras.layers.Dropout(params['dropout']))

    # If we want to also test for number of layers and shapes, that's possible
    hidden_layers(model, params, 1)

    # Then we finish again with completely standard Keras way
    model.add(keras.layers.Dense(1, activation=params['activation'], use_bias=True,
                                 kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=keras.regularizers.l1_l2(l1=params['l1'], l2=params['l2']),
                                 bias_regularizer=None))

    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['losses'],
                  metrics=['mse'])

    history = model.fit(x_train, y_train,
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        callbacks=[early_stopper(epochs=params['epochs'], mode='moderate')],
                        #callbacks=[early_stopper(epochs=params['epochs'], mode='strict')],
                        verbose=0)

    # Finally we have to make sure that history object and model are returned
    return history, model

# Then we can go ahead and set the parameters space


# Alternatively small parameters space
params = {'lr': {0.01, 0.1, 0.5},
          'l1': {0.1995262, 0.1584893, 0.1258925, 0.1000000, 0},
          'l2': {0.1995262, 0.1584893, 0.1258925, 0.1000000, 0},
          'first_neuron': {4, 8, 16, 32},
          'hidden_layers': {1, 2},
          'batch_size': {32, 64, 128},
          'epochs': {250},
          'dropout': {0, 0.1, 0.2, 0.3, 0.4},
          'optimizer': {Adam, SGD},
          'losses': [mse],
          'activation': {relu, sigmoid}}


params_final = {'lr': {0.0001},
                'l1': {0},
                'l2': {0},
                'first_neuron': {32, 128},
                'hidden_layers': {1, 2},
                'batch_size': {32},
                'epochs': {1000000},
                'dropout': {0},
                'optimizer': {Adam},
                'losses': [mse],
                'activation': {relu}}

# Run the experiment
os.chdir(path + "/Data/")

t = ta.Scan(x=x_train,
            y=y_train,
            model=build_model,
            grid_downsample=1,
            val_split=0.3,
            params=params_final,
            dataset_name='POL',
            experiment_no='2_final')

# Prediction

p = ta.Predict(t)
pred = p.predict(x_test, metric='val_loss')
MSE = np.mean((y_test - pred)**2)
print(MSE)
