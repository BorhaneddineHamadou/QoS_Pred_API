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

def predict_using_models(models, X, model='all'):
    n = len(models)  # Nombre de modèles

    # Initialisation d'une matrice pour stocker les prédictions
    predictions = np.zeros(n)

    # Record the starting time to generate predictions
    predictions_start_time = time.time()

    if model == 'esn':
      for i, model in enumerate(models):
        # Effectue les prédictions pour le modèle i
        y_pred = model.predict(np.array([X]))
        predictions[i] = y_pred
    else :

      for i, model in enumerate(models):
          # Effectue les prédictions pour le modèle i
          y_pred = model.predict(X, verbose=0)
          predictions[:, i] = np.squeeze(y_pred)

    # Record the ending time of generating predictions
    predictions_end_time = time.time()
    predictions_elapsed_time = predictions_end_time - predictions_start_time

    return predictions, predictions_elapsed_time


def rnn_forecast(models, data, input_data, selected_columns, horizon, forecasting_strategy, hyperparameters, new_variable_dataset, s=2):
    if horizon==1 and len(selected_columns)==1: # 1-step univariate
        variable = data[[selected_columns[0]]]
        variable_dataset = variable.values

        if new_variable_dataset is not None and len(new_variable_dataset) > 0:
          variable_dataset = new_variable_dataset[:]
        
        variable_dataset = np.append(variable_dataset, input_data)
        window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

        denoised_variable_dataset = moving_average(variable_dataset, window_size)

        look_back = hyperparameters["look_back"]

        # reshape into X=t and Y=t+1
        trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

        predictions = models.predict(np.array([trainX[-1]]))
        predictions = np.array(predictions)

        return predictions, variable_dataset
        
    elif horizon==1 and len(selected_columns)>1: # 1-step multivariate
        # Process each column
        processed_columns = []
        min_length = len(data)
        for i, column_name in enumerate(selected_columns):
            # Extract the column from the DataFrame
            column_data = data[[column_name]]
            
            # Convert the column to a NumPy array
            column_dataset = column_data.values
            
            if new_variable_dataset[i] is not None and len(new_variable_dataset[i]) > 0:
                column_dataset = new_variable_dataset[i]

            column_dataset = np.append(column_dataset, input_data[i])
            new_variable_dataset[i] = column_dataset[:]
            
            # Smooth the data
            window_size, slide_size = smooth_ASAP(column_dataset, resolution=50)
            
            denoised_column_dataset = moving_average(column_dataset, window_size)
            
            # Determine the minimum length
            length = len(denoised_column_dataset)
            if length < min_length:
                min_length = length
            
            # Reshape the data
            denoised_column_dataset = denoised_column_dataset.reshape((length, 1))

            processed_columns.append(denoised_column_dataset)

        # Horizontally stack the processed columns
        for i in range(len(processed_columns)):
            processed_columns[i] = processed_columns[i][:min_length]
        dataset = np.hstack(processed_columns)

        look_back = hyperparameters["look_back"]

        # convert into input/output
        X, Y = create_multivariate_dataset(dataset, look_back)

        # Reshape X in 2D Dimention
        X = np.reshape(X, (X.shape[0], -1))
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        # make predictions
        testPredict = models.predict(np.array([X[-1]]))
        testPredict = np.array(testPredict)

        return testPredict, new_variable_dataset

    elif horizon>1 and len(selected_columns)==1: # N-step univariate
        if forecasting_strategy == "Recursive":
            def rnn_recursive_strategy(model, X_row, n_steps):
                forecasts = []
                shape_0 = X_row.shape[0]
                shape_1 = X_row.shape[1]

                for i in range(n_steps):
                    X_row = np.reshape(X_row, (shape_0, 1, shape_1))
                    forecast = model.predict(X_row, verbose=0)
                    X_row.reshape(X_row.shape[2],)
                    forecasts.append(forecast[0, 0])
                    X_row = X_row.tolist()
                    X_row[0][0].append(forecast[0, 0])
                    X_row = X_row[0][0][1:]
                    X_row = np.array(X_row)
                return forecasts
                
            def rnn_make_predictions(model, X, n_steps):
                predictions = []
                row_forecasts = rnn_recursive_strategy(model, X, n_steps)
                predictions.append(row_forecasts)
                return predictions
            
            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            if new_variable_dataset is not None and len(new_variable_dataset) > 0:
                variable_dataset = new_variable_dataset[:]
            
            variable_dataset = np.append(variable_dataset, input_data)
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            look_back = hyperparameters["look_back"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            testPredict = rnn_make_predictions(models, trainX[-1], horizon)

            
            return testPredict, variable_dataset
        
        elif forecasting_strategy == "Direct":

            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values

            if new_variable_dataset is not None and len(new_variable_dataset) > 0:
                variable_dataset = new_variable_dataset[:]
            
            variable_dataset = np.append(variable_dataset, input_data)

            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            look_back = hyperparameters["look_back"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, horizon)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            # make predictions
            testPredict, predictions_elapsed_time = predict_using_models(models, trainX[-1], model='esn')
            testPredict = np.array(testPredict)

            return testPredict, variable_dataset
        
        elif forecasting_strategy == "DirRec":
            def predict_with_dirrec_models(models, testX, horizon, model_type='all'):
                predictions = []

                # Record the starting time to generate predictions
                predicting_start_time = time.time()

                for h in range(horizon):
                    model = models[h]

                    # Predict using the current model and test data
                    prediction = model.predict(np.array([testX]))
                    print("h = ", h)
                    predictions.append(prediction)

                    if model_type == 'esn':
                      testX = np.concatenate((testX, prediction), axis=1)
                    else:
                      testX = np.concatenate((testX, prediction[:, np.newaxis]), axis=-1)

                # Record the ending time of generate predictions
                predicting_end_time = time.time()
                predicting_elapsed_time = predicting_end_time - predicting_start_time

                return np.array(predictions).T, predicting_elapsed_time

            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            if new_variable_dataset is not None and len(new_variable_dataset) > 0:
                variable_dataset = new_variable_dataset[:]
            
            variable_dataset = np.append(variable_dataset, input_data)

            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            look_back = hyperparameters["look_back"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, horizon)
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            # make predictions
            testPredict, predictions_elapsed_time = predict_with_dirrec_models(models, trainX[-1], horizon, 'esn')
            testPredict = np.array(testPredict)
            testPredict = np.squeeze(testPredict)

            return testPredict, variable_dataset
        
        elif forecasting_strategy == "MIMO":
            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            if new_variable_dataset is not None and len(new_variable_dataset) > 0:
                variable_dataset = new_variable_dataset[:]
            
            variable_dataset = np.append(variable_dataset, input_data)

            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)
            
            look_back = hyperparameters["look_back"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, horizon)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            testPredict = models.predict(np.array([trainX[-1]]))
            testPredict = np.array(testPredict)

            return testPredict, variable_dataset
        
        elif forecasting_strategy == "DIRMO":
            
            def predict_using_models(models, X, s):
                n = len(models)  # Nombre de modèles

                # Initialisation d'une matrice pour stocker les prédictions
                predictions = np.zeros((X.shape[0], n*s))

                # Record the starting time to generate predictions
                predictions_start_time = time.time()

                for i, model in enumerate(models):
                    # Effectue les prédictions pour le modèle i
                    y_pred = model.predict(np.array([X]), verbose=0)
                    predictions[:, s*i:s*i+s] = np.squeeze(y_pred)

                # Record the ending time of generating predictions
                predictions_end_time = time.time()
                predictions_elapsed_time = predictions_end_time - predictions_start_time
                return predictions, predictions_elapsed_time

            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values

            if new_variable_dataset is not None and len(new_variable_dataset) > 0:
                variable_dataset = new_variable_dataset[:]
            
            variable_dataset = np.append(variable_dataset, input_data)
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            look_back = hyperparameters["look_back"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, horizon)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            # make predictions
            testPredict, predictions_elapsed_time = predict_using_models(models, trainX[-1], s)
            testPredict = np.array(testPredict)

            return testPredict, variable_dataset