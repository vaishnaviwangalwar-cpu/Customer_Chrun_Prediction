# Customer Churn Prediction

An end-to-end machine learning application that predicts the likelihood of a bank customer churning (leaving the bank), built on an Artificial Neural Network and deployed as an interactive web app.

**Live application:** [customerchrunprediction-2509.streamlit.app](https://customerchrunprediction-2509.streamlit.app)

 ![App demo](assets/demo.gif)
 
---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Pipeline](#model-pipeline)
- [Getting Started](#getting-started)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Overview

Customer churn is one of the costliest problems for subscription and account-based businesses: retaining an existing customer is consistently cheaper than acquiring a new one. This project addresses that problem directly by building a predictive model that flags customers likely to leave, so that retention efforts can be targeted before it happens.

The system is built around a feed-forward Artificial Neural Network trained on a bank customer dataset, using standard features such as credit score, geography, account balance, and activity status. The trained model is served through a Streamlit web application that returns a real-time churn probability, a clear risk classification, and supporting insights for any customer profile entered by the user.

The project covers the complete applied machine learning lifecycle: data preprocessing, model development, hyperparameter tuning, and production deployment.

---

## Key Features

- Real-time churn prediction powered by a trained neural network, with results returned instantly on user input
- A clean, guided input interface using sliders, dropdowns, and numeric fields for every customer attribute
- A visual risk indicator that classifies each prediction as High Risk or Low Risk with an associated probability score
- An automated insights panel that surfaces specific risk factors for a given customer, such as low product engagement, inactivity, short tenure, or zero balance
- Efficient model loading through Streamlit's resource caching, keeping repeated predictions fast
- A fully reproducible pipeline, from raw dataset to trained model to deployed application

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow / Keras |
| Preprocessing | scikit-learn (StandardScaler, LabelEncoder, OneHotEncoder) |
| Hyperparameter Tuning | SciKeras |
| Data Handling | pandas, NumPy |
| Web Application | Streamlit |
| Experiment Tracking | TensorBoard |
| Visualization | Matplotlib |
| Deployment | Streamlit Community Cloud |

Dependencies are pinned in `requirements.txt`:

```
tensorflow==2.21.0
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
scikeras
```

---

## Project Structure

```
Customer_Chrun_Prediction/
├── app.py                          Streamlit application (entry point)
├── Churn_Modelling.csv             Training dataset
├── experiments.ipynb               ANN model development and training
├── hyperparametertuningann.ipynb   Hyperparameter tuning experiments
├── prediction.ipynb                Standalone inference and testing
├── salaryregression.ipynb          Supplementary regression experiment
├── model.h5                        Trained ANN (Keras model)
├── scaler.pkl                      Fitted StandardScaler
├── label_encoder_gender.pkl        Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl          Fitted OneHotEncoder for Geography
├── requirements.txt                Python dependencies
├── runtime.txt                     Python version pin
└── README.md
```

---

## Dataset

The model is trained on `Churn_Modelling.csv`, a bank customer dataset containing the following fields:

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Customer's country |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Years as a customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | Whether the customer holds a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Customer's estimated salary |
| Exited | Target variable — whether the customer churned |

---

## Model Pipeline

1. **Encoding** — `Gender` is label-encoded and `Geography` is one-hot encoded to convert categorical fields into numerical form.
2. **Feature assembly** — encoded categorical features are combined with the remaining numerical features into a single input vector.
3. **Scaling** — all features are standardized using a pre-fitted `StandardScaler` to match the distribution the model was trained on.
4. **Inference** — the scaled input is passed through the trained ANN, which outputs a churn probability through a sigmoid activation.
5. **Classification** — a probability above 0.5 is classified as High Risk; below 0.5 as Low Risk.

Model development and tuning are documented across three notebooks: `experiments.ipynb` for initial architecture and training, `hyperparametertuningann.ipynb` for systematic tuning, and `prediction.ipynb` for validating inference outside the app. `salaryregression.ipynb` applies the same neural network approach to a related regression task, estimating customer salary.

---

## Getting Started

### Prerequisites

- Python 3.11
- pip

### Installation

```bash
git clone https://github.com/vaishnaviwangalwar-cpu/Customer_Chrun_Prediction.git
cd Customer_Chrun_Prediction

python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## Deployment

The application is deployed and publicly accessible on Streamlit Community Cloud:

**[customerchrunprediction-2509.streamlit.app](https://customerchrunprediction-2509.streamlit.app)**

The deployment is configured directly from this repository:

1. The repository is connected to Streamlit Community Cloud via GitHub.
2. `app.py` is set as the application entry point.
3. `requirements.txt` and `runtime.txt` define the build environment and Python version automatically on deploy.
4. The trained model (`model.h5`) and preprocessing artifacts (`scaler.pkl`, `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`) are committed to the repository, so the deployed app loads them directly with no additional setup.

Any push to the connected branch triggers an automatic redeploy.

---

## Roadmap

- Publish model evaluation metrics (accuracy, precision, recall, ROC-AUC) alongside the model
- Add batch prediction support for CSV uploads
- Integrate explainability (e.g. SHAP) to surface feature-level reasoning behind each prediction
- Containerize the application with Docker for platform-agnostic deployment
- Add a licensing file to clarify usage terms

---

## Author

**Vaishnavi Wangalwar**
GitHub: [@vaishnaviwangalwar-cpu](https://github.com/vaishnaviwangalwar-cpu)
