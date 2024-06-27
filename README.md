# ClasPy
ClasPy is a Streamlit application that allows users to compare the performance of different machine learning classifiers on various datasets.

**Deployed App:** [ClasPy ML App](https://claspy-ml.streamlit.app/)

## Setup

### Create and activate a virtual environment:
```sh
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On MacOS/Linux
```
### Install the dependencies:

#### Option 1: Using `requirements.txt`
```sh
pip install -r requirements.txt

```
#### Option 2: Install individually
```sh
pip install streamlit scikit-learn matplotlib
```

## Run 
```sh
streamlit run main.py
```

## Activate "Run on save" in Settings
Automatically updates the app when the underlying code is updated.
