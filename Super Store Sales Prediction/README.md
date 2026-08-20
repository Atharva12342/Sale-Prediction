# 🏬 Super Store Sales Prediction System

## 📌 Project Overview

The **Super Store Sales Prediction System** is a Machine Learning-based web application developed using **Python, Scikit-Learn, Pandas, NumPy, Plotly, and Streamlit**.

The main purpose of this project is to predict **supermarket outlet sales** based on different product and outlet characteristics.

The project uses a **K-Nearest Neighbors (KNN) Regression** model trained on historical supermarket sales data. The trained model is integrated into an interactive **Streamlit dashboard**, where users can enter product and store details and receive an estimated sales value.

The application also provides interactive visualizations for understanding the predicted sales and comparing estimated sales across different supermarket formats.

---

# 🎯 Project Objectives

The major objectives of this project are:

* Predict supermarket outlet sales using Machine Learning.
* Analyze product and outlet characteristics.
* Apply data preprocessing and feature engineering.
* Encode categorical variables for Machine Learning.
* Scale numerical features using StandardScaler.
* Train a KNN Regression model.
* Save the trained model using Pickle.
* Build an interactive Streamlit web application.
* Display prediction results using interactive Plotly charts.
* Compare estimated sales across different outlet types.

---

# 🚀 Key Features

### 🤖 Machine Learning

* K-Nearest Neighbors (KNN) Regression
* Feature scaling using StandardScaler
* Categorical feature encoding
* Model serialization using Pickle
* Prediction using trained Machine Learning model

### 📊 Interactive Dashboard

The Streamlit application provides:

* Product input form
* Store/outlet input form
* Sales prediction
* Prediction benchmark gauge
* Store-format comparison chart
* Item MRP metric
* Item visibility metric
* Store format information
* Store location information

### 🎨 Modern UI

The application includes:

* Custom CSS
* Glassmorphism-style cards
* Background image
* Responsive layout
* Styled tabs
* Interactive input components
* Dashboard-style presentation

---

# 📂 Project Structure

The following structure represents the project files:

```text
Super-Store-Sales-Prediction/
│
├── .ipynb_checkpoints/
│   └── Jupyter Notebook Checkpoints
│
├── app.py
│
├── feature_columns.pkl
│
├── KNN Regression.ipynb
│
├── KNN_reg_outlet_sales.xlsx
│
├── knn_model.pkl
│
├── preprocessing.pkl
│
└── scaler.pkl
```

---

# 📄 File & Folder Description

## 1. `app.py`

`app.py` is the **main Streamlit application file**.

It is responsible for creating the complete user interface and connecting the trained Machine Learning model with the dashboard.

### Main responsibilities:

* Configure the Streamlit page.
* Load the trained KNN model.
* Load the scaler.
* Load the feature columns.
* Accept user inputs.
* Perform data preprocessing.
* Encode categorical features.
* Scale input data.
* Generate sales predictions.
* Display prediction results.
* Create Plotly visualizations.
* Display the project information and developer details.

The application loads:

```text
knn_model.pkl
scaler.pkl
feature_columns.pkl
```

and uses them to generate predictions.

### Run the application:

```bash
streamlit run app.py
```

---

# 2. `KNN Regression.ipynb`

`KNN Regression.ipynb` is the **Jupyter Notebook used for Machine Learning development**.

This notebook contains the model development and training process.

### Main activities performed in the notebook:

```text
Dataset Loading
       ↓
Data Understanding
       ↓
Data Cleaning
       ↓
Missing Value Handling
       ↓
Feature Engineering
       ↓
Categorical Encoding
       ↓
Feature Scaling
       ↓
Train-Test Split
       ↓
KNN Regression Model
       ↓
Model Evaluation
       ↓
Model Saving
```

The notebook is mainly used during the **development and training phase** of the project.

After training, the required objects are saved as `.pkl` files so that the Streamlit application can use the trained model without retraining it every time.

---

# 3. `KNN_reg_outlet_sales.xlsx`

`KNN_reg_outlet_sales.xlsx` is the **dataset file** used for the project.

It contains historical supermarket product and outlet information that is used for training the Machine Learning model.

The dataset contains information related to:

### Product Features

* Item Weight
* Item Fat Content
* Item Visibility
* Item Type
* Item MRP

### Outlet Features

* Outlet Identifier
* Outlet Establishment Year
* Outlet Size
* Outlet Location Type
* Outlet Type

### Target

* Outlet Sales / Sales Revenue

The dataset is used as the primary source for training and evaluating the KNN Regression model.

---

# 4. `knn_model.pkl`

`knn_model.pkl` contains the **trained KNN Regression Machine Learning model**.

The model is saved using Python's Pickle library after the training process.

The Streamlit application loads this file using:

```python
with open("knn_model.pkl", "rb") as file:
    model = pickle.load(file)
```

The loaded model is then used to predict sales:

```python
prediction = model.predict(input_scaled)
```

### Purpose

This file allows the application to make predictions without retraining the Machine Learning model.

---

# 5. `scaler.pkl`

`scaler.pkl` contains the **StandardScaler object** used during model training.

Feature scaling is important for KNN because KNN calculates distances between data points.

The scaler transforms input features into a standardized format.

The application loads it using:

```python
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
```

The user's input is then scaled before prediction:

```python
input_scaled = scaler.transform(input_data)
```

### Purpose

Ensures that new input data is processed in the same way as the training data.

---

# 6. `feature_columns.pkl`

`feature_columns.pkl` contains the **final feature column structure** used during model training.

This is especially important because categorical variables are converted into multiple columns using encoding.

The application loads the feature columns:

```python
with open("feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)
```

The input data is then aligned with these columns:

```python
input_data = input_data.reindex(
    columns=feature_columns,
    fill_value=0
)
```

### Purpose

Ensures that the input data has exactly the same feature structure expected by the trained Machine Learning model.

---

# 7. `preprocessing.pkl`

`preprocessing.pkl` is a serialized preprocessing object created during the Machine Learning development process.

It can be used to store preprocessing-related information or transformations required during model development.

Depending on the training notebook implementation, this file can contain preprocessing configuration, transformation objects, or related preprocessing information.

### Purpose

It helps maintain consistency between the data preprocessing performed during training and the preprocessing required during prediction.

> **Note:** The current `app.py` code directly uses `scaler.pkl` and performs categorical encoding inside the application. Therefore, `preprocessing.pkl` is not directly loaded by the current Streamlit code.

---

# 8. `.ipynb_checkpoints`

`.ipynb_checkpoints` is a folder automatically created by **Jupyter Notebook**.

It contains automatically saved checkpoint versions of notebooks.

For example:

```text
.ipynb_checkpoints/
    KNN Regression-checkpoint.ipynb
```

These files are mainly useful during notebook development.

### GitHub Recommendation

You normally do **not need to upload** `.ipynb_checkpoints` to GitHub.

You can add the following to `.gitignore`:

```text
.ipynb_checkpoints/
```

---

# 🧠 Machine Learning Workflow

The complete workflow of the project is:

```text
                 Dataset
                    │
                    ▼
             Data Cleaning
                    │
                    ▼
          Missing Value Handling
                    │
                    ▼
          Feature Engineering
                    │
                    ▼
       Categorical Feature Encoding
                    │
                    ▼
             Feature Scaling
                    │
                    ▼
             Train-Test Split
                    │
                    ▼
           KNN Regression Model
                    │
                    ▼
             Model Evaluation
                    │
                    ▼
          Save Model as .pkl
                    │
                    ▼
           Streamlit Application
                    │
                    ▼
              User Input
                    │
                    ▼
             Data Preprocessing
                    │
                    ▼
            Scaled Input Data
                    │
                    ▼
             KNN Prediction
                    │
                    ▼
             Predicted Sales
                    │
                    ▼
           Interactive Dashboard
```

---

# 🔢 Input Features

The Streamlit application accepts the following input parameters.

| Feature                   | Description                            |
| ------------------------- | -------------------------------------- |
| Item Weight               | Weight of the product                  |
| Item Fat Content          | Fat category of the product            |
| Item Visibility           | Visibility of the product in the store |
| Item Type                 | Product category                       |
| Item MRP                  | Maximum Retail Price                   |
| Outlet Identifier         | Unique outlet ID                       |
| Outlet Establishment Year | Year the outlet was established        |
| Outlet Size               | Size of the outlet                     |
| Outlet Location Type      | Tier of the outlet location            |
| Outlet Type               | Type of supermarket/store              |

---

# 🏪 Supported Outlet Types

The application supports the following outlet types:

```text
Grocery Store
Supermarket Type1
Supermarket Type2
Supermarket Type3
```

---

# 📍 Supported Location Tiers

```text
Tier 1
Tier 2
Tier 3
```

---

# 📦 Supported Item Types

The application supports multiple product categories:

```text
Baking Goods
Breads
Breakfast
Canned
Dairy
Frozen Foods
Fruits and Vegetables
Hard Drinks
Health and Hygiene
Household
Meat
Others
Seafood
Snack Foods
Soft Drinks
Starchy Foods
```

---

# 🛠️ Technologies Used

| Technology           | Usage                |
| -------------------- | -------------------- |
| Python               | Programming          |
| Pandas               | Data processing      |
| NumPy                | Numerical operations |
| Scikit-Learn         | Machine Learning     |
| KNN Regression       | Sales prediction     |
| StandardScaler       | Feature scaling      |
| Pickle               | Model serialization  |
| Streamlit            | Web application      |
| Plotly Express       | Interactive charts   |
| Plotly Graph Objects | Gauge visualization  |
| Jupyter Notebook     | Model development    |
| Excel                | Dataset storage      |

---

# 📦 Python Libraries

Install the required libraries using:

```bash
pip install streamlit pandas numpy scikit-learn plotly openpyxl
```

Or create a `requirements.txt` file:

```text
streamlit
pandas
numpy
scikit-learn
plotly
openpyxl
```

Then install:

```bash
pip install -r requirements.txt
```

---

# ⚙️ Installation & Setup

## Step 1: Clone the Repository

```bash
git clone https://github.com/darshanbhor2006/Super-Store-Sales-Prediction.git
```

## Step 2: Open the Project

```bash
cd Super-Store-Sales-Prediction
```

## Step 3: Create Virtual Environment

```bash
python -m venv venv
```

## Step 4: Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

## Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Run the following command:

```bash
streamlit run app.py
```

After successful execution, Streamlit will provide a local address such as:

```text
http://localhost:8501
```

Open the address in your browser.

---

# 🔮 How to Use the Application

### Step 1

Open the **Prediction Form** tab.

### Step 2

Enter product information:

* Item Type
* Item MRP
* Item Weight
* Fat Content
* Visibility Score

### Step 3

Enter outlet information:

* Outlet Identifier
* Outlet Type
* Outlet Size
* Location Tier
* Establishment Year

### Step 4

Click:

```text
🔮 Predict Super Store Sales
```

### Step 5

The application generates the predicted sales value.

### Step 6

View the interactive charts:

* Predicted Sales vs Benchmark
* Sales Comparison Across Store Formats

---

# 📊 Dashboard

The dashboard contains two major sections:

## 📊 Prediction Form

Users can enter product and store details and generate a sales prediction.

## ℹ️ About Us

The About section provides:

* Project description
* Machine Learning workflow
* Features
* Technology stack
* Business applications
* Developer information
* Contact information

---

# 📈 Visualizations

The application uses **Plotly** to provide interactive visualizations.

### 1. Sales Benchmark Gauge

The gauge displays the predicted sales and compares it against a benchmark.

### 2. Store Format Comparison

A bar chart compares estimated sales across different outlet formats.

```text
Grocery Store
Supermarket Type1
Supermarket Type2
Supermarket Type3
```

---

# 💼 Business Applications

The project can be useful for retail businesses for:

* 📦 Inventory planning
* 📈 Sales forecasting
* 🏪 Outlet performance analysis
* 💰 Revenue estimation
* 🎯 Product planning
* 🛒 Retail decision making
* 📊 Business analytics
* 📋 Demand planning

---

# ⭐ Advantages

* Easy-to-use interface
* Interactive Streamlit dashboard
* Fast prediction
* Machine Learning-based forecasting
* Interactive Plotly charts
* Multiple outlet categories
* Product-level prediction
* Reusable trained model
* Suitable for academic and portfolio projects

---

# 🔮 Future Enhancements

The project can be improved by adding:

* Random Forest Regression
* XGBoost Regression
* Gradient Boosting
* Model comparison
* Hyperparameter tuning
* Cross-validation
* Model performance dashboard
* Historical sales trend analysis
* Downloadable prediction reports
* User authentication
* Database integration
* Cloud deployment
* Automated model retraining
* SHAP-based model explainability

---

# 👨‍💻 Developer

## Darshan Bhor

**Role:** Machine Learning Developer

### Skills

* Python
* Machine Learning
* Scikit-Learn
* Pandas
* NumPy
* Data Visualization
* Streamlit
* Plotly

---

# 📬 Contact

📧 **Email:** darshanbhor2006@gmail.com

💼 **LinkedIn:**
https://www.linkedin.com/in/darshan-bhor

🐙 **GitHub:**
https://github.com/darshanbhor2006

---

# 📌 Project Highlights

```text
🏬 Super Store Sales Prediction System

🤖 Machine Learning
📊 KNN Regression
🐍 Python
🌐 Streamlit
📈 Plotly
🐼 Pandas
🔢 NumPy
📦 Pickle
📁 Jupyter Notebook
💰 Sales Prediction
🏪 Retail Analytics
```

---

# 📜 License

This project is developed for **educational, learning, and portfolio purposes**.

© 2026 **Darshan Bhor**. All Rights Reserved.

---

# ⭐ Support

If you find this project useful, please consider giving the repository a ⭐ **Star** on GitHub.
