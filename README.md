# 🎯 Customer Segmentation

A machine learning project that segments customers into six groups using K-Means clustering based on customer demographics, income, spending, purchasing activity, and recency.

## 📌 Project Overview

This project uses K-Means clustering to identify different customer segments.

The model uses the following features:

* Age
* Income
* Total Spending
* Number of Web Purchases
* Number of Store Purchases
* Number of Web Visits per Month
* Recency

The features are standardized using `StandardScaler` before applying K-Means clustering.

The trained model is used in a Streamlit application to predict the customer segment for a new customer.

## 📂 Project Structure

```text
Customer-segmentation/
│
├── app.py
├── customer_segmentation.csv
├── kmeans_model.pkl
├── scaler.pkl
├── cluster_summary.csv
├── requirements.txt
├── README.md
└── customer_segmentation.ipynb
```

## ⚙️ Requirements

* Python 3.10 or newer
* pip
* Virtual environment recommended

The project uses:

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Seaborn
* Streamlit
* Jupyter

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/SAQIB-KHAN-25/Customer-segmentation.git
cd Customer-segmentation
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it in Command Prompt:

```bash
.venv\Scripts\activate
```

For PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Streamlit recommends using a virtual environment to isolate project dependencies.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Streamlit installation

```bash
streamlit --version
```

## 🤖 Model Files

The application requires the following trained files:

```text
kmeans_model.pkl
scaler.pkl
cluster_summary.csv
```

These files are generated from the customer segmentation notebook.

The K-Means model and cluster summary should be generated from the same training run so that their cluster IDs remain consistent.

## 🧠 Training the Model

Open the Jupyter notebook:

```bash
jupyter notebook
```

Run the notebook from beginning to end.

The training process:

1. Loads `customer_segmentation.csv`
2. Cleans the data
3. Creates customer features
4. Calculates `Total_Spending`
5. Standardizes the features using `StandardScaler`
6. Trains a six-cluster K-Means model
7. Generates cluster statistics
8. Saves the trained model
9. Saves the scaler
10. Saves `cluster_summary.csv`

The K-Means model uses a fixed random state to make the training reproducible.

The trained files are:

```text
kmeans_model.pkl
scaler.pkl
cluster_summary.csv
```

## ▶️ Run the Streamlit Application

Make sure the virtual environment is activated and you are in the project root directory.

Run:

```bash
streamlit run app.py
```

Streamlit will start a local server and normally open the application in your browser.

If the `streamlit` command is not recognized, use:

```bash
python -m streamlit run app.py
```

## 🖥️ Using the Application

Enter the following customer information:

* Age
* Income
* Total Spending
* Number of Web Purchases
* Number of Store Purchases
* Number of Web Visits
* Recency

Click:

```text
🎯 Predict Segment
```

The application displays:

* Predicted cluster
* Customer segment name
* Customer profile
* Recommended business strategy
* Distance from each cluster
* Customer summary

## 📊 Customer Segments

The current model contains six clusters:

| Cluster | Segment                         |
| ------: | ------------------------------- |
|       0 | Inactive Low-Value Customers    |
|       1 | Active Valuable Customers       |
|       2 | High-Income Premium Customers   |
|       3 | Low-Value Active Customers      |
|       4 | Highest-Value Premium Customers |
|       5 | Low-Value Inactive Customers    |

These labels are based on the generated cluster statistics in `cluster_summary.csv`.

## 🔍 Model and Summary Validation

The application verifies that the cluster IDs in the trained K-Means model match the cluster IDs in `cluster_summary.csv`.

The expected cluster IDs are:

```text
0, 1, 2, 3, 4, 5
```

If the model and summary contain different cluster IDs, the application stops and displays an error instead of showing potentially incorrect segment information.

## 🛑 Stop the Application

To stop the Streamlit server, press:

```text
Ctrl + C
```

in the terminal.

## 🔄 Rebuilding the Model

If the customer data or feature engineering changes, regenerate the model and cluster summary together by running the notebook from beginning to end.

Do not manually modify `kmeans_model.pkl` or `cluster_summary.csv`.

After retraining, verify that these files are updated together:

```text
kmeans_model.pkl
scaler.pkl
cluster_summary.csv
```

## 🛠️ Troubleshooting

### `ModuleNotFoundError`

Run:

```bash
pip install -r requirements.txt
```

### `FileNotFoundError: kmeans_model.pkl`

Make sure `kmeans_model.pkl` is present in the project root.

### `FileNotFoundError: scaler.pkl`

Make sure `scaler.pkl` is present in the project root.

### `FileNotFoundError: cluster_summary.csv`

Run the training notebook to generate the cluster summary.

### Streamlit command not found

Try:

```bash
python -m streamlit run app.py
```

## 📄 License

This project is intended for educational and portfolio purposes.
