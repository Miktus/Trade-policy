import talos as ta
from keras.layers import Dropout, Dense
from keras.models import Sequential
from talos.model.normalizers import lr_normalizer

from keras.optimizers import Adam, Nadam
from keras.activations import relu, elu
from keras.losses import mse

# Comment


x, y = ta.datasets.iris()
type(x)
x.shape

p = {'activation': [relu, elu], 'lr': (0.1, 10, 10),
     'optimizer': [Nadam, Adam],
     'losses': [mse],
     # 'hidden_layers': [0, 1, 2],
     'batch_size': [20],
     'epochs': [10]}

# then we can go ahead and set the parameter space
p = {'lr': (0.5, 5, 10),
     'l1': [0.1, 0.2],
     'l2': [0.1, 0.2],
     'first_neuron': [4, 8, 16, 32, 64],
     'hidden_layers': [0, 1, 2],
     'batch_size': (2, 3, 4),
     'epochs': [150],
     'dropout': (0, 0.5, 5),
     'weight_regulizer': [None],
     'emb_output_dims': [None],
     'shape': ['brick', 'long_funnel'],
     'optimizer': ['Adam', 'Nadam', 'SGD'],
     'losses': ['MSE', 'MAE'],
     'activation': ['tanh', 'relu', 'sigmoid']}


def iris_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Dense(32, input_dim=4, activation=params['activation']))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['losses'])
    # metrics=['acc'])

    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    validation_data=[x_val, y_val],
                    verbose=0)

    return out, model


scan_object = ta.Scan(x, y, model=iris_model, params=p, grid_downsample=0.1)
help(ta.Scan)

scan_object.details
