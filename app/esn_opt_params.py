TRIALS = 2
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

def predict_using_models(models, X, model='all'):
    n = len(models)  # Nombre de modèles

    # Initialisation d'une matrice pour stocker les prédictions
    predictions = np.zeros((X.shape[0], n))

    # Record the starting time to generate predictions
    predictions_start_time = time.time()

    if model == 'esn':
      for i, model in enumerate(models):
        # Effectue les prédictions pour le modèle i
        y_pred = model.predict(X)
        predictions[:, i] = np.squeeze(y_pred)
    else :

      for i, model in enumerate(models):
          # Effectue les prédictions pour le modèle i
          y_pred = model.predict(X, verbose=0)
          predictions[:, i] = np.squeeze(y_pred)

    # Record the ending time of generating predictions
    predictions_end_time = time.time()
    predictions_elapsed_time = predictions_end_time - predictions_start_time

    return predictions, predictions_elapsed_time

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

def optimize_esn_parameters(data, selected_columns, horizon, forecasting_strategy, s=2):
    if horizon==1 and len(selected_columns)==1: # 1-step univariate
        variable = data[[selected_columns[0]]]
        variable_dataset = variable.values
        window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

        denoised_variable_dataset = moving_average(variable_dataset, window_size)

        # split into train and test sets
        train_size = int(len(denoised_variable_dataset) * 0.9)
        test_size = len(denoised_variable_dataset) - train_size
        train, test = denoised_variable_dataset[0:train_size], denoised_variable_dataset[train_size:]
        # Define the objective function for Optuna optimization
        def objective(trial):
            # Sample hyperparameters to search

            n_reservoir = trial.suggest_int('n_reservoir', 10, 1000)   # -
            sparsity = trial.suggest_categorical('sparsity', [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])   # -
            spectral_radius = trial.suggest_categorical('spectral_radius', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0])   # - spectral radius of W
            noise = trial.suggest_categorical('noise', [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])       # - Noise Set
            look_back = trial.suggest_int('look_back', 10, 500)

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(train, look_back, 1)
            validX, validY = create_multistep_dataset(test, look_back, 1)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
            validX = np.reshape(validX, (validX.shape[0], validX.shape[1]))

            # Build and fit the ESN model

            model = ESN(n_inputs = look_back,
                        n_outputs = 1,
                        n_reservoir = n_reservoir,
                        sparsity=sparsity,
                        random_state=1234,
                        spectral_radius=spectral_radius,
                        noise = noise,
                        teacher_scaling = 10)

            # Train and test our model
            pred_train = model.fit(trainX, trainY)
            predictions = model.predict(validX)
            predictions = np.array(predictions)

            # Evaluate the model on the validation set
            val_loss = np.sqrt(mean_squared_error(predictions, validY))

            return val_loss
        # Create the Optuna study
        study = optuna.create_study(direction='minimize')

        # Record the starting time to generate predictions
        start_time = time.time()

        # Run the optimization
        study.optimize(objective, n_trials=TRIALS)

        # Record the ending time
        end_time = time.time()
        elapsed_time = end_time - start_time

        print('done')
        print("ESN HyperParameters Tuning Elapsed Time : %.5f" % (elapsed_time), "seconds")

        # Print the best parameters and corresponding loss
        best_params = study.best_params
        best_loss = study.best_value
        
        return best_params, best_loss
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

        # Define the objective function for Optuna optimization
        def objective(trial):
            # Sample hyperparameters to search

            n_reservoir = trial.suggest_int('n_reservoir', 10, 1000)   # -
            sparsity = trial.suggest_categorical('sparsity', [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])   # -
            spectral_radius = trial.suggest_categorical('spectral_radius', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0])   # - spectral radius of W
            noise = trial.suggest_categorical('noise', [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])       # - Noise Set
            look_back = trial.suggest_int('look_back', 10, 500)

            # convert into input/output
            X, Y = create_multivariate_dataset(dataset, look_back)

            # Reshape X in 2D Dimention
            X = np.reshape(X, (X.shape[0], -1))

            test_size = int(len(X) * 0.10)

            # Divisez manuellement X et Y en ensembles d'entraînement et de test
            trainX, trainY, testX, testY = X[:-test_size], Y[:-test_size], X[-test_size:], Y[-test_size:]
            print(trainX.shape, trainY.shape)

            model = ESN(n_inputs = trainX.shape[1],
                    n_outputs = Y.shape[1],
                    n_reservoir = n_reservoir,
                    sparsity=sparsity,
                    random_state=1234,
                    spectral_radius=spectral_radius,
                    noise = noise,
                    silent = False,
            )

            pred_train = model.fit(trainX, trainY)

            # make predictions
            testPredict = model.predict(testX)
            testPredict = np.array(testPredict)

            column_RMSE = []
            for i in range(dataset.shape[1]):
                true_values = testY[:, i]
                predicted_values = testPredict[:len(testY), i]
                rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
                column_RMSE.append(rmse)

            testRMSE = np.sqrt(mean_squared_error(testY, testPredict[:len(testY), :]))

            return np.sum(column_RMSE)


        # Create the Optuna study
        study = optuna.create_study(direction='minimize')

        # Record the starting time to generate predictions
        start_time = time.time()

        # Run the optimization
        study.optimize(objective, n_trials=TRIALS)

        # Record the ending time
        end_time = time.time()
        elapsed_time = end_time - start_time

        print('done')
        print("ESN HyperParameters Tuning Elapsed Time : %.5f" % (elapsed_time), "seconds")

        # Print the best parameters and corresponding loss
        best_params = study.best_params
        best_loss = study.best_value
        return best_params, best_loss
    elif horizon>1 and len(selected_columns)==1: # N-step univariate
        if forecasting_strategy == "Recursive":
            def esn_recursive_strategy(model, X_row, n_steps):
                forecasts = []

                for i in range(n_steps):
                    forecast = model.predict(np.array([X_row]))
                    forecasts.append(forecast[0, 0])
                    X_row = X_row.tolist()
                    X_row.append(forecast[0, 0])
                    X_row = X_row[1:]
                    X_row = np.array(X_row)
                return forecasts
            def esn_make_predictions(model, X, n_steps):
                predictions = []
                for i in range(len(X)):
                    row_forecasts = esn_recursive_strategy(model, X[i, :], n_steps)
                    predictions.append(row_forecasts)
                return predictions
            
            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            # split into train and test sets
            train_size = int(len(denoised_variable_dataset) * 0.9)
            test_size = len(denoised_variable_dataset) - train_size
            train, test = denoised_variable_dataset[0:train_size], denoised_variable_dataset[train_size:]

            # Define the objective function for Optuna optimization
            def objective(trial):
                # Sample hyperparameters to search

                n_reservoir = trial.suggest_int('n_reservoir', 10, 1000)   # -
                sparsity = trial.suggest_categorical('sparsity', [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])   # -
                spectral_radius = trial.suggest_categorical('spectral_radius', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0])   # - spectral radius of W
                noise = trial.suggest_categorical('noise', [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])       # - Noise Set
                look_back = trial.suggest_int('look_back', 10, 500)

                # reshape into X=t and Y=t+1
                trainX, trainY = create_multistep_dataset(train, look_back, 1)
                testX, testY = create_multistep_dataset(test, look_back, 1)

                # reshape input to be [samples, time steps, features]
                trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
                testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))

                # Build and fit the ESN model

                model = ESN(n_inputs = look_back,
                            n_outputs = 1,
                            n_reservoir = n_reservoir,
                            sparsity=sparsity,
                            random_state=1234,
                            spectral_radius=spectral_radius,
                            noise = noise,
                            teacher_scaling = 10)

                # Train and test our model
                pred_train = model.fit(trainX, trainY)
                
                predictions_start_time = time.time()

                testPredict = esn_make_predictions(model, testX, horizon)

                # Record the ending time of generating predictions
                predictions_end_time = time.time()
                predictions_elapsed_time = predictions_end_time - predictions_start_time

                testPredict = np.array(testPredict)
                _, new_testY = create_multistep_dataset(test, look_back, horizon)

                testRMSE = np.sqrt(mean_squared_error(new_testY, testPredict[:len(new_testY), :]))
                testMAE = mean_absolute_error(new_testY, testPredict[:len(new_testY), :])

                # Evaluate the model on the validation set
                val_loss = testRMSE

                return val_loss
            # Create the Optuna study
            study = optuna.create_study(direction='minimize')

            # Record the starting time to generate predictions
            start_time = time.time()

            # Run the optimization
            study.optimize(objective, n_trials=TRIALS)

            # Record the ending time
            end_time = time.time()
            elapsed_time = end_time - start_time

            print('done')
            print("ESN HyperParameters Tuning Elapsed Time : %.5f" % (elapsed_time), "seconds")

            # Print the best parameters and corresponding loss
            best_params = study.best_params
            best_loss = study.best_value
            
            return best_params, best_loss
        
        elif forecasting_strategy == "Direct":
            def direct_multistep_strategy_op_params(trainX, trainY, look_back, n_reservoir, sparsity, spectral_radius, noise):
                n = trainY.shape[1]  # Nombre de colonnes de Y (horizons de prévision)
                models = []  # Liste pour stocker les modèles entraînés

                total_training_elapsed_time = 0
                for i in range(n):
                    y = trainY[:, i]  # Sélectionne la colonne i de Y

                    # Crée et entraîne le modèle pour l'horizon de prévision i

                    model = ESN(n_inputs = look_back,
                        n_outputs = 1,
                        n_reservoir = n_reservoir,
                        sparsity=sparsity,
                        random_state=1234,
                        spectral_radius=spectral_radius,
                        noise = noise,
                        silent = False,
                        teacher_scaling = 10,
                    )
                    # Record the starting time to training the model
                    training_start_time = time.time()

                    pred_train = model.fit(trainX, y)

                    # Record the ending time of training the model
                    training_end_time = time.time()
                    training_elapsed_time = training_end_time - training_start_time
                    total_training_elapsed_time += training_elapsed_time
                    models.append(model)  # Ajoute le modèle à la liste

                    print("Model Done !")
                return models, total_training_elapsed_time
            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            # split into train and test sets
            train_size = int(len(denoised_variable_dataset) * 0.9)
            test_size = len(denoised_variable_dataset) - train_size
            train, test = denoised_variable_dataset[0:train_size], denoised_variable_dataset[train_size:]

            # Define the objective function for Optuna optimization
            def objective(trial):
                # Sample hyperparameters to search

                n_reservoir = trial.suggest_int('n_reservoir', 10, 1000)   # -
                sparsity = trial.suggest_categorical('sparsity', [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])   # -
                spectral_radius = trial.suggest_categorical('spectral_radius', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0])   # - spectral radius of W
                noise = trial.suggest_categorical('noise', [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])       # - Noise Set
                look_back = trial.suggest_int('look_back', 10, 500)

                # reshape into X=t and Y=t+1
                trainX, trainY = create_multistep_dataset(train, look_back, horizon)
                validX, validY = create_multistep_dataset(test, look_back, horizon)

                # reshape input to be [samples, time steps, features]
                trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
                validX = np.reshape(validX, (validX.shape[0], validX.shape[1]))

                models, training_elapsed_time = direct_multistep_strategy_op_params(trainX, trainY, look_back, n_reservoir, sparsity, spectral_radius, noise)

                # make predictions
                testPredict, predictions_elapsed_time = predict_using_models(models, validX, model='esn')
                testPredict = np.array(testPredict)

                testRMSE = np.sqrt(mean_squared_error(validY, testPredict[:len(validY), :]))

                return testRMSE


            # Create the Optuna study
            study = optuna.create_study(direction='minimize')

            # Record the starting time to generate predictions
            start_time = time.time()

            # Run the optimization
            study.optimize(objective, n_trials=TRIALS)

            # Record the ending time
            end_time = time.time()
            elapsed_time = end_time - start_time

            print('done')
            print("ESN HyperParameters Tuning Elapsed Time : %.5f" % (elapsed_time), "seconds")

            # Print the best parameters and corresponding loss
            best_params = study.best_params
            best_loss = study.best_value
            return best_params, best_loss
        
        elif forecasting_strategy == "DirRec":
            def create_dirrec_esn_model(n_inputs, n_reservoir, sparsity, spectral_radius, noise):
                model = ESN(n_inputs = n_inputs,
                n_outputs = 1,
                n_reservoir = n_reservoir,
                sparsity=sparsity,
                random_state=1234,
                spectral_radius=spectral_radius,
                noise = noise,
                silent = False,
                teacher_scaling = 10,
                )

                return model
            def DirRecESNStrategy(trainX, trainY, horizon, n_reservoir, sparsity, spectral_radius, noise):

                models = []
                input_set = []
                total_training_elapsed_time = 0

                for h in range(horizon):
                    model = create_dirrec_esn_model(trainX.shape[1], n_reservoir, sparsity, spectral_radius, noise)

                    # Record the starting time to training the model
                    training_start_time = time.time()

                    # Train the model with the current input data and target
                    model.fit(trainX, trainY[:, h])


                    # Record the ending time of training the model
                    training_end_time = time.time()
                    training_elapsed_time = training_end_time - training_start_time
                    total_training_elapsed_time += training_elapsed_time

                    models.append(model)

                    # Update the input set with the current model's prediction
                    predictions = model.predict(trainX)
                    trainX = np.concatenate((trainX, predictions), axis=1)

                return models, total_training_elapsed_time
            def predict_with_dirrec_esn_models(models, testX, model_type='all'):
                predictions = []

                # Record the starting time to generate predictions
                predicting_start_time = time.time()

                for h in range(horizon):
                    model = models[h]

                    # Predict using the current model and test data
                    prediction = model.predict(testX)
                    predictions.append(prediction)

                    testX = np.concatenate((testX, prediction), axis=1)

                # Record the ending time of generate predictions
                predicting_end_time = time.time()
                predicting_elapsed_time = predicting_end_time - predicting_start_time

                return np.array(predictions).T, predicting_elapsed_time
            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            # split into train and test sets
            train_size = int(len(denoised_variable_dataset) * 0.9)
            test_size = len(denoised_variable_dataset) - train_size
            train, test = denoised_variable_dataset[0:train_size], denoised_variable_dataset[train_size:]

            # Define the objective function for Optuna optimization
            def objective(trial):
                # Sample hyperparameters to search

                n_reservoir = trial.suggest_int('n_reservoir', 10, 1000)   # -
                sparsity = trial.suggest_categorical('sparsity', [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])   # -
                spectral_radius = trial.suggest_categorical('spectral_radius', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0])   # - spectral radius of W
                noise = trial.suggest_categorical('noise', [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])       # - Noise Set
                look_back = trial.suggest_int('look_back', 10, 500)

                # reshape into X=t and Y=t+1
                trainX, trainY = create_multistep_dataset(train, look_back, horizon)
                validX, validY = create_multistep_dataset(test, look_back, horizon)

                models, training_elapsed_time = DirRecESNStrategy(trainX, trainY, horizon, n_reservoir, sparsity, spectral_radius, noise)

                # make predictions
                testPredict, predictions_elapsed_time = predict_with_dirrec_esn_models(models, validX, 'esn')
                testPredict = np.array(testPredict)
                testPredict = np.squeeze(testPredict)

                testRMSE = np.sqrt(mean_squared_error(validY, testPredict))

                return testRMSE

            # Create the Optuna study
            study = optuna.create_study(direction='minimize')

            # Record the starting time to generate predictions
            start_time = time.time()

            # Run the optimization
            study.optimize(objective, n_trials=TRIALS)

            # Record the ending time
            end_time = time.time()
            elapsed_time = end_time - start_time

            print('done')
            print("ESN HyperParameters Tuning Elapsed Time : %.5f" % (elapsed_time), "seconds")

            # Print the best parameters and corresponding loss
            best_params = study.best_params
            best_loss = study.best_value
            return best_params, best_loss
        
        elif forecasting_strategy == "MIMO":
            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            # split into train and test sets
            train_size = int(len(denoised_variable_dataset) * 0.9)
            test_size = len(denoised_variable_dataset) - train_size
            train, test = denoised_variable_dataset[0:train_size], denoised_variable_dataset[train_size:]
            
            # Define the objective function for Optuna optimization
            def objective(trial):
                # Sample hyperparameters to search

                n_reservoir = trial.suggest_int('n_reservoir', 10, 1000)   # -
                sparsity = trial.suggest_categorical('sparsity', [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])   # -
                spectral_radius = trial.suggest_categorical('spectral_radius', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0])   # - spectral radius of W
                noise = trial.suggest_categorical('noise', [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])       # - Noise Set
                look_back = trial.suggest_int('look_back', 10, 500)

                # reshape into X=t and Y=t+1
                trainX, trainY = create_multistep_dataset(train, look_back, horizon)
                validX, validY = create_multistep_dataset(test, look_back, horizon)

                # reshape input to be [samples, time steps, features]
                trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
                validX = np.reshape(validX, (validX.shape[0], validX.shape[1]))

                model = ESN(n_inputs = look_back,
                        n_outputs = horizon,
                        n_reservoir = n_reservoir,
                        sparsity=sparsity,
                        random_state=1234,
                        spectral_radius=spectral_radius,
                        noise = noise,
                        silent = False,
                        teacher_scaling = 10,
                )

                pred_train = model.fit(trainX, trainY)

                # make predictions
                testPredict = model.predict(validX)
                testPredict = np.array(testPredict)

                testRMSE = np.sqrt(mean_squared_error(validY, testPredict[:len(validY), :]))

                return testRMSE


            # Create the Optuna study
            study = optuna.create_study(direction='minimize')

            # Record the starting time to generate predictions
            start_time = time.time()

            # Run the optimization
            study.optimize(objective, n_trials=TRIALS)

            # Record the ending time
            end_time = time.time()
            elapsed_time = end_time - start_time

            print('done')
            print("ESN HyperParameters Tuning Elapsed Time : %.5f" % (elapsed_time), "seconds")

            # Print the best parameters and corresponding loss
            best_params = study.best_params
            best_loss = study.best_value
            return best_params, best_loss
        
        elif forecasting_strategy == "DIRMO":
            def esn_dirmo_strategy(X, Y, s, look_back, n_reservoir, sparsity, spectral_radius, noise):
                n = Y.shape[1]  # Nombre de colonnes de Y (horizons de prévision)
                number_of_models = int(n/s)
                models = []  # Liste pour stocker les modèles entraînés
                total_training_elapsed_time = 0

                for i in range(number_of_models):
                    y = Y[:, i:i+s]  # Sélectionne la colonne i de Y

                    # Crée et entraîne le modèle pour l'horizon de prédiction i
                    model = ESN(n_inputs = look_back,
                        n_outputs = s,
                        n_reservoir = n_reservoir,
                        sparsity=sparsity,
                        random_state=1234,
                        spectral_radius=spectral_radius,
                        noise = noise,
                        silent = False,
                    )


                    # Record the starting time to training the model
                    training_start_time = time.time()

                    pred_train = model.fit(X, y)

                    # Record the ending time of training the model
                    training_end_time = time.time()
                    training_elapsed_time = training_end_time - training_start_time
                    total_training_elapsed_time += training_elapsed_time

                    models.append(model)  # Ajoute le modèle à la liste

                    print("Model Done !")
                print("All Models Done !")

                return models, total_training_elapsed_time
            
            def esn_predict_using_models(models, X, s):
                n = len(models)  # Nombre de modèles

                # Initialisation d'une matrice pour stocker les prédictions
                predictions = np.zeros((X.shape[0], n*s))

                # Record the starting time to generate predictions
                predictions_start_time = time.time()

                for i, model in enumerate(models):
                    # Effectue les prédictions pour le modèle i
                    y_pred = model.predict(X)
                    predictions[:, s*i:s*i+s] = np.squeeze(y_pred)

                # Record the ending time of generating predictions
                predictions_end_time = time.time()
                predictions_elapsed_time = predictions_end_time - predictions_start_time
                return predictions, predictions_elapsed_time
            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            # split into train and test sets
            train_size = int(len(denoised_variable_dataset) * 0.9)
            test_size = len(denoised_variable_dataset) - train_size
            train, test = denoised_variable_dataset[0:train_size], denoised_variable_dataset[train_size:]

            # Define the objective function for Optuna optimization
            def objective(trial):
                # Sample hyperparameters to search

                n_reservoir = trial.suggest_int('n_reservoir', 10, 1000)   # -
                sparsity = trial.suggest_categorical('sparsity', [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])   # -
                spectral_radius = trial.suggest_categorical('spectral_radius', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0])   # - spectral radius of W
                noise = trial.suggest_categorical('noise', [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])       # - Noise Set
                look_back = trial.suggest_int('look_back', 10, 500)

                # reshape into X=t and Y=t+1
                trainX, trainY = create_multistep_dataset(train, look_back, horizon)
                validX, validY = create_multistep_dataset(test, look_back, horizon)

                # reshape input to be [samples, time steps, features]
                trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
                validX = np.reshape(validX, (validX.shape[0], validX.shape[1]))

                models, training_elapsed_time = esn_dirmo_strategy(trainX, trainY, s, look_back, n_reservoir, sparsity, spectral_radius, noise)

                # make predictions
                testPredict, predictions_elapsed_time = esn_predict_using_models(models, validX, s)
                testPredict = np.array(testPredict)

                testRMSE = np.sqrt(mean_squared_error(validY, testPredict[:len(validY), :]))

                return testRMSE


            # Create the Optuna study
            study = optuna.create_study(direction='minimize')

            # Record the starting time to generate predictions
            start_time = time.time()

            # Run the optimization
            study.optimize(objective, n_trials=TRIALS)

            # Record the ending time
            end_time = time.time()
            elapsed_time = end_time - start_time

            print('done')
            print("ESN HyperParameters Tuning Elapsed Time : %.5f" % (elapsed_time), "seconds")

            # Print the best parameters and corresponding loss
            best_params = study.best_params
            best_loss = study.best_value
            return best_params, best_loss
