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
from .esn_forecast import *
from .rnn_forecast import *


# Functions for ARIMA Model

# multistep sarima forecast
def sarima_multistep_forecast(history, config, window_size, n_steps):
    order, sorder, trend = config
    new_hist = history[:]
    yhat = []
    total_train_time = 0
    total_prediction_time = 0
    # define model
    for i in range(n_steps):

      model = SARIMAX(new_hist[-window_size:], order=order, seasonal_order=sorder, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)

      # Record the starting time to train the model
      training_start_time = time.time()

      # fit model
      model_fit = model.fit(disp=False)

      # Record the end time from training the model
      training_end_time = time.time()
      elapsed_training_time = training_end_time - training_start_time
      total_train_time += elapsed_training_time

      # make multistep forecast
      # yhat = model_fit.forecast(steps=n_steps)

      # Record the starting time to generate predictions
      predictions_start_time = time.time()

      prediction = model_fit.predict(start=len(history[-window_size:]), end=len(history[-window_size:]))

      # Record the end time from generate predictions
      predictions_end_time = time.time()
      elapsed_predicting_time = predictions_end_time - predictions_start_time
      total_prediction_time += elapsed_predicting_time

      yhat = np.append(yhat, prediction)
      new_hist = np.append(new_hist, prediction)
      new_hist = new_hist[1:]
    return yhat, total_train_time, total_prediction_time

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

def sarima_forecast(data, input_data, selected_columns, horizon, hyperparameters, new_variable_dataset):
  
      # define hyperparameters
      p = hyperparameters["p"]
      d = hyperparameters["d"]
      q = hyperparameters["q"]
      P = 0
      D = 0
      Q = 0
      m = 0
      trend = 'c'
      
      window_length = hyperparameters["window_length"]

      order = (p, d, q)
      seasonal_order = (P, D, Q, m)

      cfg = (order, seasonal_order, trend)

      variable = data[[selected_columns[0]]]

      variable_dataset = variable.values

      if new_variable_dataset is not None and len(new_variable_dataset) > 0:
         variable_dataset = new_variable_dataset[:]
      
      variable_dataset = np.append(variable_dataset, input_data)
      window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

      print("Window Size: ", window_size)
      denoised_variable_dataset = moving_average(variable_dataset, window_size)
      
      predictions = []
      ys = []
      # seed history with training dataset
      history = []
      history.extend(denoised_variable_dataset)

      # step over each time-step in the test set
      # for i = 0
      # fit model and make forecast for history
      yhat,total_train_time, total_prediction_time = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
      
      # store forecast in list of predictions
      predictions.append(yhat)
      
      return predictions, variable_dataset

def pers_forecast(data, input_data, selected_columns, horizon, new_variable_dataset):

    if horizon==1 and len(selected_columns)>1: # 1-step multivariate
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
            
            testPredict = [input_data]

        return testPredict, new_variable_dataset
        
    else :

        variable = data[[selected_columns[0]]]

        variable_dataset = variable.values

        if new_variable_dataset is not None and len(new_variable_dataset) > 0:
            variable_dataset = new_variable_dataset[:]
        
        variable_dataset = np.append(variable_dataset, input_data)
        window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

        print("Window Size: ", window_size)
        denoised_variable_dataset = moving_average(variable_dataset, window_size)
        
        predictions = [input_data * horizon]
        
        return predictions, variable_dataset

def forecast(models, processed_dataset, input_data, selected_columns, selected_horizon, forecasting_strategy, forecasting_model, hyperparameters, new_variable_dataset, s=2):
        if forecasting_model == "ARIMA":
            predictions, new_variable_dataset = sarima_forecast(processed_dataset, input_data, selected_columns, selected_horizon, hyperparameters, new_variable_dataset)
        elif forecasting_model == "ESN":
            predictions, new_variable_dataset = esn_forecast(models, processed_dataset, input_data, selected_columns, selected_horizon, forecasting_strategy, hyperparameters, new_variable_dataset, s)
        elif forecasting_model == "RNN" or forecasting_model == "LSTM" or forecasting_model == "GRU":
            predictions, new_variable_dataset = rnn_forecast(models, processed_dataset, input_data, selected_columns, selected_horizon, forecasting_strategy, hyperparameters, new_variable_dataset, s)
        elif forecasting_model == "Persistence":
            predictions, new_variable_dataset = pers_forecast(processed_dataset, input_data, selected_columns, selected_horizon, new_variable_dataset)
        
        return predictions, new_variable_dataset
