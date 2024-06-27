# ClasPy

ClasPy is a Streamlit application that allows users to compare the performance of different machine learning classifiers on various datasets.

**Deployed App:** [ClasPy ML App](https://claspy-ml-app.streamlit.app/)

## Setup

### Prerequisites

- Python 3.x

### Installation 

1. **Clone the repository**:
    ```sh
    git clone https://github.com/lisek75/claspy_ml_app.git
    cd claspy_ml_app
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    source .venv/bin/activate  # On MacOS/Linux
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Run 

1. **Start the Streamlit application**:
    ```sh
    streamlit run main.py
    ```

2. **Enable "Run on Save" in Streamlit Settings**:

    To automatically update the app when the underlying code is updated:
    - Run the Streamlit app.
    - In the browser, click on the three dots (⋮) in the top-right corner.
    - Select "Settings".
    - Toggle the "Run on save" option to enable it.

## Usage

- Open your browser and navigate to the local Streamlit URL (usually `http://localhost:8501`).
- Select the desired ML task and dataset using the sidebar options.
- View the performance comparison and analysis results.
