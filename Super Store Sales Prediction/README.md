**# 🏬 Super Store Sales Prediction

A **Machine Learning-based web application** that predicts supermarket outlet sales using product and outlet characteristics.

## 📌 Overview

This project uses **K-Nearest Neighbors (KNN) Regression** to estimate supermarket sales based on historical sales data. The trained model is integrated into an interactive **Streamlit** dashboard where users can enter product and outlet details and receive real-time sales predictions.

## 🚀 Features

* Sales prediction using KNN Regression
* Data preprocessing and feature encoding
* Feature scaling using StandardScaler
* Interactive Streamlit dashboard
* Plotly-based data visualizations
* Store-format sales comparison
* Saved ML model for fast predictions

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy**
* **Scikit-Learn**
* **KNN Regression**
* **Streamlit**
* **Plotly**
* **Jupyter Notebook**
* **Pickle**

## 📂 Project Structure

```text
Super-Store-Sales-Prediction/
│
├── app.py
├── KNN Regression.ipynb
├── KNN_reg_outlet_sales.xlsx
├── knn_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── preprocessing.pkl
└── requirements.txt
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/darshanbhor2006/Super-Store-Sales-Prediction.git
cd Super-Store-Sales-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at the local Streamlit address.

## 📊 Input Features

The model uses features such as:

* Item Weight
* Item Fat Content
* Item Visibility
* Item Type
* Item MRP
* Outlet Identifier
* Outlet Size
* Outlet Location
* Outlet Type
* Outlet Establishment Year

## 🎯 Applications

This system can support:

* Sales forecasting
* Inventory planning
* Revenue estimation
* Outlet performance analysis
* Retail decision-making

