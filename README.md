# Edge QoS Predictor API

<strong>Edge QoS Predictor</strong> is an API that utilizes advanced time series forecasting models to predict QoS values in edge environments. This user-friendly application is the primary interface for interacting with the Edge QoS Predictor API, which powers these essential tasks and provides real-time predictions into edge network's performance.


## Getting Started

Follow these steps to launch the project on a Linux system. Please note that the commands may vary on Windows or macOS.

### Prerequisites

Make sure you have Python installed on your system.

### Installation

1. Optionally, you can create a virtual execution environment:

    ```bash
    python -m venv .
    ```

2. Activate the environment:

    ```bash
    source bin/activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

4. Launch the server:

    ```bash
    uvicorn app.main:app --reload
    ```

5. Launch the frontend app:

    ```bash
    streamlit run app/streamlit-app.py
    ```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

