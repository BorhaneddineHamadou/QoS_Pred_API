from .hyperparams_tuning import *
from .model_training import *
from .forecasting import *
import pathlib
from typing import Optional
from fastapi import FastAPI, Query, FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import pandas as pd
from pydantic import BaseModel
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = FastAPI()

BASE_DIR = pathlib.Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR.parent / "dataset"

# Global variable for the processed dataset
processed_dataset = None
selected_columns = None
selected_horizon = None
forecasting_strategy = None
forecasting_model = None
hyperparameters = None
model = None
total_training_time = None
new_dataset = None
all_predictions = []

@app.post("/")
async def upload_dataset(file: UploadFile):

    global processed_dataset

    # Check if the file format is CSV or Excel
    if file.filename.endswith((".csv", ".xlsx")):
        # Create the datasets directory if it doesn't exist
        os.makedirs(DATASET_DIR, exist_ok=True)

        # Save the uploaded file to the datasets directory
        dataset_path = os.path.join(DATASET_DIR, file.filename)
        with open(dataset_path, "wb") as f:
            f.write(file.file.read())

        # Read the dataset and identify real or integer columns
        try:
            df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_excel(dataset_path)
            processed_dataset = df  # Store the processed dataset
            numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
            return JSONResponse(content={"columns": numeric_columns}, status_code=200)
        except Exception as e:
            return JSONResponse(content={"error": f"Error processing the dataset: {str(e)}"}, status_code=400)
    else:
        return JSONResponse(content={"error": "Unsupported file format. Please upload a CSV or Excel file."}, status_code=400)

# Columns selection

class Columns(BaseModel):
    selected_columns: list

@app.post("/columns")
async def columns(selected_columns_input:Columns):
    global processed_dataset
    global selected_columns
    if processed_dataset.empty:
        return HTTPException(status_code=400, detail="Dataset has not been uploaded and processed.")
    
    # Update the global selected columns with the input
    selected_columns = selected_columns_input.selected_columns
    # Return Selected Columns
    return JSONResponse(content={"columns": selected_columns}, status_code=200)

# Forecasting horizon selection
class PredictionHorizon(BaseModel):
    horizon: int

@app.post('/horizon')
async def set_prediction_horizon(prediction_horizon: PredictionHorizon):
    # Extract the selected prediction horizon from the request
    global processed_dataset
    global selected_columns
    if processed_dataset.empty:
        return HTTPException(status_code=400, detail="Dataset has not been uploaded and processed.")
    if selected_columns is not None and len(selected_columns) == 0:
        return HTTPException(status_code=400, detail="You didn't select any columns yet.")
    global selected_horizon
    selected_horizon = prediction_horizon.horizon

    # You can use the selected_horizon value as needed in your server logic
    # For this example, we'll just return it as a response
    return JSONResponse(content={"horizon": selected_horizon}, status_code=200)


# Multistep Forecasting Strategy Selection
class PredictionStrategy(BaseModel):
    strategy: str

@app.post('/strategy')
async def set_prediction_horizon(prediction_strategy: PredictionStrategy):
    # Extract the selected prediction strategy from the request
    global processed_dataset
    global selected_columns
    global selected_horizon

    if processed_dataset.empty:
        return HTTPException(status_code=400, detail="Dataset has not been uploaded and processed.")
    if selected_columns is not None and len(selected_columns) == 0:
        return HTTPException(status_code=400, detail="You didn't select any columns yet.")
    if selected_horizon == 1 :
        return HTTPException(status_code=400, detail="You have to select a strategy only if you have multistep forecasting.")
    global forecasting_strategy
    forecasting_strategy = prediction_strategy.strategy

    # For this example, we'll just return it as a response
    return JSONResponse(content={"strategy": forecasting_strategy}, status_code=200)

class PredictionModel(BaseModel):
    model: str

@app.post('/model')
async def set_prediction_horizon(prediction_model: PredictionModel):
    # Extract the selected prediction strategy from the request
    global processed_dataset
    global selected_columns
    global selected_horizon
    global forecasting_strategy

    if processed_dataset.empty:
        return HTTPException(status_code=400, detail="Dataset has not been uploaded and processed.")
    if selected_columns is not None and len(selected_columns) == 0:
        return HTTPException(status_code=400, detail="You didn't select any columns yet.")
    
    global forecasting_model
    forecasting_model = prediction_model.model
    global hyperparameters
    # Load the JSON file
    current_directory = os.path.dirname(os.path.abspath(__file__))
    jon_file = os.path.join(current_directory, "params.json")
    with open(jon_file, 'r') as json_file:
        data = json.load(json_file)

    if forecasting_model == 'ARIMA':
        hyperparameters = data['ARIMA']
    elif forecasting_model == 'Persistence':
        hyperparameters = None
    else:
        if selected_columns is not None and len(selected_columns) > 1:  # Multivariate
            hyperparameters = data[forecasting_model]['multivariate']
        else:  # Univariate
            if selected_columns[0] not in data[forecasting_model]['univariate']:
                hyperparameters = data['default']
            else:
                if str(selected_horizon) in data[forecasting_model]['univariate'][selected_columns[0]]:
                    hyperparameters = data[forecasting_model]['univariate'][selected_columns[0]][str(selected_horizon)]
                    if int(selected_horizon) > 1 :
                        hyperparameters = hyperparameters[forecasting_strategy]
                else:
                    # Find the closest larger entry to selected_horizon
                    horizon_values = [int(key) for key in data[forecasting_model]['univariate'][selected_columns[0]].keys()]
                    closest_horizon = str(max(filter(lambda x: x < int(selected_horizon), horizon_values)))
                    hyperparameters = data[forecasting_model]['univariate'][selected_columns[0]][str(closest_horizon)]
                    if int(selected_horizon) > 1 and int(closest_horizon) > 1:
                        hyperparameters = hyperparameters[forecasting_strategy]

    # For this example, we'll just return it as a response
    return JSONResponse(content={"model": forecasting_model, "hyperparameters": hyperparameters}, status_code=200)

@app.get('/hypopt')
async def hypopt():
    # Extract the selected prediction strategy from the request
    global processed_dataset
    global selected_columns
    global selected_horizon
    global forecasting_strategy
    global forecasting_model

    if processed_dataset.empty:
        return HTTPException(status_code=400, detail="Dataset has not been uploaded and processed.")
    if selected_columns is None or len(selected_columns) == 0:
        return HTTPException(status_code=400, detail="You didn't select any columns yet.")
    if selected_horizon is None :
        return HTTPException(status_code=400, detail="You have to select a forecasting horizon.")
    if forecasting_model is None :
        return HTTPException(status_code=400, detail="You have to select a forecasting model.")
    
    global hyperparameters
    hyperparameters, best_error = hyperparams_tuning(processed_dataset, selected_columns, selected_horizon, forecasting_strategy, forecasting_model)

    # For this example, we'll just return it as a response
    return JSONResponse(content={"hyperparameters": hyperparameters, "Best Error": best_error}, status_code=200)

@app.get('/train')
async def train():
    # Extract the selected prediction strategy from the request
    global processed_dataset
    global selected_columns
    global selected_horizon
    global forecasting_strategy
    global forecasting_model
    global hyperparameters

    if processed_dataset.empty:
        return HTTPException(status_code=400, detail="Dataset has not been uploaded and processed.")
    if selected_columns is None or len(selected_columns) == 0:
        return HTTPException(status_code=400, detail="You didn't select any columns yet.")
    if selected_horizon is None :
        return HTTPException(status_code=400, detail="You have to select a forecasting horizon.")
    if forecasting_model is None :
        return HTTPException(status_code=400, detail="You have to select a forecasting model.")
    if hyperparameters is None :
        return HTTPException(status_code=400, detail="There are no values for hyperparameters.")
    
    global model
    global total_training_time
    model, total_training_time = model_training(processed_dataset, selected_columns, selected_horizon, forecasting_strategy, forecasting_model, hyperparameters)
    # For this example, we'll just return it as a response
    return JSONResponse(content={"Model": str(model), "Total Training Time": f"{total_training_time} secondes"}, status_code=200)

class InputData(BaseModel):
    input_data: str

@app.get("/predict")
async def predict_endpoint(input_data:InputData):
    input_data = input_data.input_data
    global processed_dataset
    global selected_columns
    global forecasting_model
    global selected_horizon
    global hyperparameters
    global model
    global all_predictions
    global new_dataset

    if input_data == 'stop':
        if len(all_predictions) == 0 or new_dataset is None :
            return HTTPException(status_code=400, detail=f"Start making some predictions first.")
        l = len(all_predictions)
        actual = np.array(new_dataset)
        if len(selected_columns) > 1:
            actual = actual[:, -l:]
            predicted = np.array(all_predictions).flatten()
        elif selected_horizon > 1 :
            actual = actual[-l:]
            predicted = np.array(all_predictions).flatten()[:len(actual)]
        else :
            actual = actual[-l:]
            predicted = np.array(all_predictions).flatten()
        actual = actual.flatten()
        
        
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        return JSONResponse(content={"RMSE": rmse, "MAE": mae}, status_code=200)
    else:
        try:
            # Attempt to parse input_data as a float
            input_data = [float(x.strip()) for x in input_data.split(',') if x.strip()]
        except ValueError:
            return HTTPException(status_code=400, detail="Please enter numerical data.")
    
    if not isinstance(input_data, list):
        input_data = [input_data]
    
    if not is_numeric_list(input_data):
        return HTTPException(status_code=400, detail="Please enter valid data.")
    if processed_dataset is None or processed_dataset.empty :
        return HTTPException(status_code=400, detail="Dataset has not been uploaded and processed.")
    if selected_columns is None or len(selected_columns) == 0:
        return HTTPException(status_code=400, detail="You didn't select any columns yet.")
    if forecasting_model is None :
        return HTTPException(status_code=400, detail="You have to select a forecasting model.")
    if selected_horizon is None :
        return HTTPException(status_code=400, detail="You have to select a forecasting horizon.")
    if forecasting_model != "Persistence" and hyperparameters is None :
        return HTTPException(status_code=400, detail="There are no values for hyperparameters.")
    if forecasting_model != "Persistence" and forecasting_model != "ARIMA" and model is None :
        return HTTPException(status_code=400, detail="There is no trained model.")
    if len(selected_columns) != len(input_data) :
        return HTTPException(status_code=400, detail="Please enter an exact number of values as the selected columns.")
    
    if new_dataset is None and len(selected_columns)>1:
        new_dataset = [None] * len(selected_columns)
    
    predictions, new_dataset = forecast(model, processed_dataset, input_data, selected_columns, selected_horizon, forecasting_strategy, forecasting_model, hyperparameters, new_dataset)
    all_predictions.append(predictions)
    return JSONResponse(content={f"Prediction": str(predictions)}, status_code=200)

def is_numeric_list(lst):
    for item in lst:
        if not isinstance(item, (int, float)):
            return False
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


# @app.get('/')
# def read_index(q:Optional[str] = None): # q is a URL parameter (/?q=0.5)
#     query = q or "hello world"
#     return {"query":query}

