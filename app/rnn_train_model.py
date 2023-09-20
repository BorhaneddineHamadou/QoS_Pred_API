# Importing functions and classes we'll use

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dropout, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from warnings import catch_warnings
from warnings import filterwarnings
import keras
import sys
import scipy.stats
import json
import numpy.fft
import time
from decimal import Decimal
import math
from math import sqrt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler

from .ASAP import *
from .esn import *


def create_multistep_dataset(data, n_input, n_out=1):
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end]
            X.append(x_input)
            y.append(data[in_end:out_end])
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)


def create_multivariate_dataset(sequences, look_back):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + look_back
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def process_multivariate_column(data, column_name, min_length):
    # Extract the column from the DataFrame
    column_data = data[[column_name]]
    
    # Convert the column to a NumPy array
    column_dataset = column_data.values
    
    # Smooth the data
    window_size, slide_size = smooth_ASAP(column_dataset, resolution=50)
    print(f"Window Size for {column_name}: {window_size}")
    
    denoised_column_dataset = moving_average(column_dataset, window_size)
    
    # Determine the minimum length
    length = len(denoised_column_dataset)
    if length < min_length:
        min_length = length
    
    # Reshape the data
    denoised_column_dataset = denoised_column_dataset.reshape((length, 1))
    return denoised_column_dataset, min_length

def train_rnn_model(data, forecasting_model, selected_columns, horizon, forecasting_strategy, hyperparameters, s=2):
    if horizon==1 and len(selected_columns)==1: # 1-step univariate
        variable = data[[selected_columns[0]]]
        variable_dataset = variable.values
        window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

        denoised_variable_dataset = moving_average(variable_dataset, window_size)
        
        # define hyperparameters
        look_back = hyperparameters["look_back"]
        num_hidden_layers = hyperparameters["num_hidden_layers"]

        learning_rate = hyperparameters["learning_rate"]
        batch_size = hyperparameters["batch_size"]
        epochs = hyperparameters["epochs"]

        # reshape into X=t and Y=t+1
        trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

        # Crée et entraîne le modèle pour l'horizon de prévision i
        model = Sequential()
        for i in range(num_hidden_layers):
            num_units = hyperparameters[f'rnn_units_layer_{i}']
            return_sequences = (i < num_hidden_layers - 1)
            if forecasting_model=="RNN":
                model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
            elif forecasting_model=="LSTM":
                model.add(LSTM(units=num_units, return_sequences=return_sequences))
            elif forecasting_model=="GRU":
                model.add(GRU(units=num_units, return_sequences=return_sequences))
        model.add(Dense(1))
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        # Record the starting time to train the model
        training_start_time = time.time()

        # Train our model
        model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Record the ending time
        training_end_time = time.time()
        training_elapsed_time = training_end_time - training_start_time
        
        return model, training_elapsed_time

    elif horizon==1 and len(selected_columns)>1: # 1-step multivariate
        # Process each column
        processed_columns = []
        min_length = len(data)
        for column_name in selected_columns:
            processed_column, min_length = process_multivariate_column(data, column_name, min_length)
            processed_columns.append(processed_column)

        # Horizontally stack the processed columns
        for i in range(len(processed_columns)):
            processed_columns[i] = processed_columns[i][:min_length]
        dataset = np.hstack(processed_columns)

        # define hyperparameters
        look_back = hyperparameters["look_back"]
        num_hidden_layers = hyperparameters["num_hidden_layers"]

        learning_rate = hyperparameters["learning_rate"]
        batch_size = hyperparameters["batch_size"]
        epochs = hyperparameters["epochs"]

        # reshape into X=t and Y=t+1
        trainX, trainY = create_multivariate_dataset(dataset, look_back)

        # Reshape X in 2D Dimention
        trainX = np.reshape(trainX, (trainX.shape[0], -1))

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

        # Crée et entraîne le modèle pour l'horizon de prévision i
        model = Sequential()
        for i in range(num_hidden_layers):
            num_units = hyperparameters[f'rnn_units_layer_{i}']
            return_sequences = (i < num_hidden_layers - 1)
            if forecasting_model=="RNN":
                model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
            elif forecasting_model=="LSTM":
                model.add(LSTM(units=num_units, return_sequences=return_sequences))
            elif forecasting_model=="GRU":
                model.add(GRU(units=num_units, return_sequences=return_sequences))
        model.add(Dense(trainY.shape[1]))
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        # Record the starting time to train the model
        training_start_time = time.time()

        # Train our model
        model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Record the ending time
        training_end_time = time.time()
        training_elapsed_time = training_end_time - training_start_time
        
        return model, training_elapsed_time

    elif horizon>1 and len(selected_columns)==1: # N-step univariate
        if forecasting_strategy == "Recursive":

            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            # define hyperparameters
            look_back = hyperparameters["look_back"]
            num_hidden_layers = hyperparameters["num_hidden_layers"]

            learning_rate = hyperparameters["learning_rate"]
            batch_size = hyperparameters["batch_size"]
            epochs = hyperparameters["epochs"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            # Crée et entraîne le modèle pour l'horizon de prévision i
            model = Sequential()
            for i in range(num_hidden_layers):
                num_units = hyperparameters[f'rnn_units_layer_{i}']
                return_sequences = (i < num_hidden_layers - 1)
                if forecasting_model=="RNN":
                    model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                elif forecasting_model=="LSTM":
                    model.add(LSTM(units=num_units, return_sequences=return_sequences))
                elif forecasting_model=="GRU":
                    model.add(GRU(units=num_units, return_sequences=return_sequences))
            model.add(Dense(1))
            optimizer = keras.optimizers.Adam(lr=learning_rate)
            model.compile(loss='mean_squared_error', optimizer=optimizer)

            # Record the starting time to train the model
            training_start_time = time.time()

            # Train our model
            model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Record the ending time
            training_end_time = time.time()
            training_elapsed_time = training_end_time - training_start_time

            return model, training_elapsed_time

        elif forecasting_strategy == "Direct":

            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            look_back = hyperparameters["look_back"]
            num_hidden_layers = hyperparameters["num_hidden_layers"]

            learning_rate = hyperparameters["learning_rate"]
            batch_size = hyperparameters["batch_size"]
            epochs = hyperparameters["epochs"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, horizon)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            n = trainY.shape[1]
            models = []

            total_training_elapsed_time = 0
            for j in range(n):

                y = trainY[:, j]  # Sélectionne la colonne j de Y

                # Crée et entraîne le modèle pour l'horizon de prévision i
                model = Sequential()
                for i in range(num_hidden_layers):
                    num_units = hyperparameters[f'rnn_units_layer_{i}']
                    return_sequences = (i < num_hidden_layers - 1)
                    if forecasting_model=="RNN":
                        model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model=="LSTM":
                        model.add(LSTM(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model=="GRU":
                        model.add(GRU(units=num_units, return_sequences=return_sequences))
                model.add(Dense(1))
                optimizer = keras.optimizers.Adam(lr=learning_rate)
                model.compile(loss='mean_squared_error', optimizer=optimizer)

                # Record the starting time to train the model
                training_start_time = time.time()

                # Train our model
                model.fit(trainX, y, epochs=epochs, batch_size=batch_size, verbose=0)
                
                # Record the ending time
                training_end_time = time.time()
                training_elapsed_time = training_end_time - training_start_time
                total_training_elapsed_time += training_elapsed_time
                
                models.append(model)  # Ajoute le modèle à la liste
                print("Model Done !")
            print("All Models Done !")

            return models, total_training_elapsed_time

        elif forecasting_strategy == "DirRec":

            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            # define hyperparameters
            look_back = hyperparameters["look_back"]
            num_hidden_layers = hyperparameters["num_hidden_layers"]

            learning_rate = hyperparameters["learning_rate"]
            batch_size = hyperparameters["batch_size"]
            epochs = hyperparameters["epochs"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, horizon)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            n = trainY.shape[1]
            models = []

            total_training_elapsed_time = 0
            for h in range(horizon):

                # Crée et entraîne le modèle pour l'horizon de prévision i
                model = Sequential()
                for i in range(num_hidden_layers):
                    num_units = hyperparameters[f'rnn_units_layer_{i}']
                    return_sequences = (i < num_hidden_layers - 1)
                    if forecasting_model=="RNN":
                        model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model=="LSTM":
                        model.add(LSTM(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model=="GRU":
                        model.add(GRU(units=num_units, return_sequences=return_sequences))
                model.add(Dense(1))
                optimizer = keras.optimizers.Adam(lr=learning_rate)
                model.compile(loss='mean_squared_error', optimizer=optimizer)

                # Record the starting time to train the model
                training_start_time = time.time()

                # Train our model
                model.fit(trainX, trainY[:, h], epochs=epochs, batch_size=batch_size, verbose=0)
                
                # Record the ending time
                training_end_time = time.time()
                training_elapsed_time = training_end_time - training_start_time
                total_training_elapsed_time += training_elapsed_time
                
                models.append(model)  # Ajoute le modèle à la liste

                # Update the input set with the current model's prediction
                predictions = model.predict(trainX)
                trainX = np.concatenate((trainX, predictions[:, np.newaxis]), axis=-1)

                print("Model Done !")
            print("All Models Done !")
                
            return models, total_training_elapsed_time

        elif forecasting_strategy == "MIMO":
            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            # define hyperparameters
            look_back = hyperparameters["look_back"]
            num_hidden_layers = hyperparameters["num_hidden_layers"]

            learning_rate = hyperparameters["learning_rate"]
            batch_size = hyperparameters["batch_size"]
            epochs = hyperparameters["epochs"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, horizon)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            # Crée et entraîne le modèle pour l'horizon de prévision i
            model = Sequential()
            for i in range(num_hidden_layers):
                num_units = hyperparameters[f'rnn_units_layer_{i}']
                return_sequences = (i < num_hidden_layers - 1)
                if forecasting_model=="RNN":
                    model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                elif forecasting_model=="LSTM":
                    model.add(LSTM(units=num_units, return_sequences=return_sequences))
                elif forecasting_model=="GRU":
                    model.add(GRU(units=num_units, return_sequences=return_sequences))
            model.add(Dense(horizon))
            optimizer = keras.optimizers.Adam(lr=learning_rate)
            model.compile(loss='mean_squared_error', optimizer=optimizer)

            # Record the starting time to train the model
            training_start_time = time.time()

            # Train our model
            model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Record the ending time
            training_end_time = time.time()
            training_elapsed_time = training_end_time - training_start_time

            return model, training_elapsed_time

        elif forecasting_strategy == "DIRMO":

            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            # define hyperparameters
            look_back = hyperparameters["look_back"]
            num_hidden_layers = hyperparameters["num_hidden_layers"]

            learning_rate = hyperparameters["learning_rate"]
            batch_size = hyperparameters["batch_size"]
            epochs = hyperparameters["epochs"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, horizon)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            
            n = trainY.shape[1]  # Nombre de colonnes de Y (horizons de prévision)
            number_of_models = int(n/s)
            models = []

            total_training_elapsed_time = 0
            for j in range(number_of_models):
                y = trainY[:, j:j+s]
                # Crée et entraîne le modèle pour l'horizon de prévision j
                model = Sequential()
                for i in range(num_hidden_layers):
                    num_units = hyperparameters[f'rnn_units_layer_{i}']
                    return_sequences = (i < num_hidden_layers - 1)
                    if forecasting_model=="RNN":
                        model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model=="LSTM":
                        model.add(LSTM(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model=="GRU":
                        model.add(GRU(units=num_units, return_sequences=return_sequences))
                model.add(Dense(s))
                optimizer = keras.optimizers.Adam(lr=learning_rate)
                model.compile(loss='mean_squared_error', optimizer=optimizer)


                # Record the starting time to train the model
                training_start_time = time.time()

                # Train our model
                model.fit(trainX, y, epochs=epochs, batch_size=batch_size, verbose=0)
                
                # Record the ending time
                training_end_time = time.time()
                training_elapsed_time = training_end_time - training_start_time
                total_training_elapsed_time += training_elapsed_time
                
                models.append(model)  # Ajoute le modèle à la liste

                print("Model Done !")
            print("All Models Done !")
            
            return models, total_training_elapsed_time