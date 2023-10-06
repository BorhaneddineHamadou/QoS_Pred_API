import streamlit as st
import requests
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

# Define the FastAPI API endpoint URL
API_URL = "http://127.0.0.1:8000/"


current_directory = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_directory, "images/logo.png")
# Set the page configuration
st.set_page_config(
    page_title="Edge QoS Predictor",
    page_icon=image_path,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to modify the background color
st.markdown(
    """
    <style>
    /* Content page */
    .css-fg4pbf {
        background-color: #0E1117; /* Page background color */
        color: #FFFFFF;
    }
    /* Sidebar */
    .css-vk3wp9 {
        background-color: #262730; /* Sidebar background color */
        color: #FFFFFF;
    }
    /* Sidebar On mobile screens */
    .css-18z3ox0{
        background-color: #262730; /* Sidebar background color */
        color: #FFFFFF;
    }
    /* Navigation color */
    .css-ue6h4q, .st-cf{
    color: #FFFFFF;
    }
    /* Title on the sidebar */
    .css-vk3wp9 h1, .css-18z3ox0 h1, h1, h3{
        color: #FFFFFF;
    }
    .css-18ni7ap{
        background: none;
    }

    /*The info box color, in the side bar*/
    .st-al{
        color: #FFFFFF;
        border: 2px solid #FF6F31;
    }
    .css-164nlkn{
        display: none;
    }
    /*Select box*/
    .st-c4{
      background: none;
    }
    .st-cf, .css-ue6h4q{
        color: inherit;
    } 
    /*Button color*/
    .stButton, .css-7ym5gk{
        color: #0E1117;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.image(image_path)
    st.title("Edge QoS Predictor")
    st.info("""
            "Edge QoS Predictor" is a user-friendly application that utilizes 
            advanced time series forecasting models to predict QoS values in edge environments.
            This application is the primary interface for interacting with the 
            Edge QoS Predictor API, which powers these essential 
            tasks and provides real-time predictions into edge network's performance.
""")

# Streamlit app title
st.title("Edge QoS Predictor API Interaction")

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'columns' not in st.session_state:
    st.session_state.columns = []
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []
if 'selected_horizon' not in st.session_state:
    st.session_state.selected_horizon = 0
if 'selected_strategy' not in st.session_state:
    st.session_state.selected_strategy = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'default_hyperparams' not in st.session_state:
    st.session_state.default_hyperparams = None
if 'predict' not in st.session_state:
    st.session_state.predict = None
if 'train_model' not in st.session_state:
    st.session_state.train_model = False


    


# Upload a file
st.session_state.uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if st.session_state.uploaded_file is not None:
    # Send the uploaded file to the FastAPI API
    response = requests.post(API_URL, files={"file": st.session_state.uploaded_file})

    if response.status_code == 200:
        result = response.json()
        st.session_state.columns = result.get("columns", [])
        st.success("File uploaded and processed successfully!")
        st.write("Numeric columns in the dataset:")
        st.write(st.session_state.columns)

# Step 2: Select Columns for Prediction
def select_columns():
    st.subheader("Step 2: Select Columns for Prediction")
    selected_columns = st.multiselect("Select target columns", st.session_state.columns)
    
    if st.button("Select Columns") or len(st.session_state.selected_columns) != 0 :
        response = requests.post(f"{API_URL}columns", json={"selected_columns": selected_columns})
        if response.status_code == 200:
            server_response = response.json()
            server_columns = server_response.get('columns', None)
            st.write(f"Selected Columns: {server_columns}")
            st.session_state.selected_columns = server_columns
            if len(st.session_state.selected_columns) > 1:
                st.success("You've selected more than 1 column, so you will perform multivariate forecasting!")
            else:
                st.success("You've selected only 1 column, so you will perform univariate forecasting!")
            st.write("Your target columns in the dataset:")
            st.write(st.session_state.selected_columns)

# 3rd Step: Select Forecasting Horizon
def select_horizon():
    st.subheader("Step 3: Select Forecasting Horizon")
    prediction_horizon = st.number_input("Select Prediction Horizon (1-30)", min_value=1, max_value=(1 if len(st.session_state.selected_columns) > 1 else 30), value=(1 if st.session_state.selected_horizon == 0 else st.session_state.selected_horizon))

    if st.button("Select Horizon") or st.session_state.selected_horizon != 0 :
        response = requests.post(f"{API_URL}horizon", json={"horizon": prediction_horizon})
        if response.status_code == 200:
            server_response = response.json()
            server_horizon = server_response.get('horizon', None)
            st.write(f"Selected Horizon: {server_horizon}")
            st.session_state.selected_horizon = server_horizon
            # Create a graphical representation of forecasting options
            
            
            # Determine which square to color based on the scenario
            if st.session_state.selected_horizon == 1 and len(st.session_state.selected_columns) == 1:
                st.image(os.path.join(current_directory, "images/1step_univariate.png"))
            elif st.session_state.selected_horizon == 1 and len(st.session_state.selected_columns) > 1:
                st.image(os.path.join(current_directory, "images/1step_multivariate.png"))
            elif st.session_state.selected_horizon > 1 and len(st.session_state.selected_columns) == 1:
                st.image(os.path.join(current_directory, "images/Nstep_univariate.png"))
            elif st.session_state.selected_horizon > 1 and len(st.session_state.selected_columns) > 1:
                st.image(os.path.join(current_directory, "images/Nstep_multivariate.png"))

            if st.session_state.selected_horizon > 1:
                select_strategy()  # Display the strategy selection step
            else:
                select_model()  # Display step 4 (or 5) if horizon is selected in step 3
        else:
            st.error("Error communicating with the server.")

# 4th Step: Select Strategy (if applicable)
def select_strategy():
    st.subheader("Step 4: Select Forecasting Strategy")
    strategy_options = ["Recursive", "Direct", "DirRec", "MIMO", "DIRMO"]
    selected_strategy = st.selectbox("Select a strategy", strategy_options)

    if st.button("Select Strategy") or st.session_state.selected_strategy != None :
        response = requests.post(f"{API_URL}strategy", json={"strategy": selected_strategy})
        if response.status_code == 200:
            server_response = response.json()
            st.write(server_response)
            st.success(f"Great! You've selected the {server_response.get('strategy')} strategy.")
            st.session_state.selected_strategy = server_response.get('strategy', None)
            select_model()  # Display step 4 (or 5) if horizon is selected in step 3

# 4th Step: Select Model
def select_model():
    model_options = ["Persistence", "ARIMA", "RNN", "LSTM", "GRU", "ESN"]
    if len(st.session_state.selected_columns) > 1:
        model_options.remove("ARIMA")
    if st.session_state.selected_horizon > 1:
        st.subheader("Step 5: Select Forecasting Model")
        if st.session_state.selected_strategy != "Recursive":
            model_options.remove("ARIMA")
    else :
        st.subheader("Step 4: Select Forecasting Model")

    
    selected_model = st.selectbox("Select a forecasting model", model_options)

    if st.button("Select Model") or st.session_state.selected_model != None:
        response = requests.post(f"{API_URL}model", json={"model": selected_model})
        if response.status_code == 200:
            server_response = response.json()
            st.write(server_response)
            st.success(f"Perfect! You've selected the {server_response.get('model')} model.")
            st.session_state.selected_model = server_response.get('model', None)
            if st.session_state.selected_model == "Persistence" :
                st.write("""
                    You've chosen a model that doesn't require training, so you can proceed directly with making predictions.
                """)
                prediction()
            else :
                st.session_state.default_hyperparams = server_response.get('hyperparameters')
                st.write("Default Hyperparameters:")
                st.write(st.session_state.default_hyperparams)
                # Create a radio button to select the action
                selected_action = st.radio("Select an action:", [":orange[Train model with default hyperparams]", ":orange[Optimize hyperparams]"])
                # Check the selected action and call the corresponding function
                if selected_action == ":orange[Train model with default hyperparams]":
                    # Call the train_model() function
                    train_model()
                elif selected_action == ":orange[Optimize hyperparams]":
                    # Call the optimize_hyperparameters() function
                    optimize_hyperparameters()


def train_model():
    if st.session_state.selected_model == "ARIMA" or st.session_state.selected_model == "Persistence" :
        st.write("""
            You've chosen a model that doesn't require training, so you can proceed directly with making predictions.
        """)
        prediction()
    else :
        if st.button("Train Model", key="train_button"):
            response = requests.get(f"{API_URL}train")

            if response.status_code == 200:
                st.success("Perfect ! The Model is Trained, you can perform real-time predictions now.")
                st.json(response.json())
                st.session_state.train_model = True
            else:
                st.error("Error training the model. Please try again.")
                return None
        if st.session_state.train_model:
            prediction()

def optimize_hyperparameters():
    if st.session_state.selected_model == "Persistence" :
        st.write("""
            You've chosen a model that doesn't require training, so you can proceed directly with making predictions.
        """)
        prediction()

    else:
        if st.button("Lunch Hyperparams Tuning", key="lunch_optimize_button"):
            response = requests.get(f"{API_URL}hypopt")
                
            if response.status_code == 200:
                best_hyperparams = response.json()
                st.write("Best Hyperparameters Found:")
                st.json(best_hyperparams)
                st.success("Hyperparameter optimization completed successfully.")
                train_model()
            else:
                st.error("Error optimizing hyperparameters. Please try again.")

def prediction():

    input_data = st.text_input("Enter input_data:")
    if st.button("Predict") or st.session_state.predict :
        st.session_state.predict = True
        response = requests.get(f"{API_URL}predict", json={"input_data": input_data})
        if response.status_code == 200:
            predictions = response.json()
            if input_data == 'stop':
                st.json(predictions)
            else:
                st.write("Prediction")
                st.write(predictions)
        else:
            st.error("Error while predicting. Please try again.")
# Check if the user has uploaded a file
if st.session_state.uploaded_file is not None:
    select_columns()  # Display step 2 if a file is uploaded

if st.session_state.selected_columns:
    select_horizon()  # Display step 3 if columns are selected in step 2

