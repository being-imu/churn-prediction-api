# Churn Prediction API (Flask)

## Overview
This project implements an end-to-end machine learning pipeline for **customer churn prediction** using a Gradient Boosting model.  
The trained model is exposed through a **Flask REST API** to provide real-time churn predictions based on customer attributes.

The goal of this project is to demonstrate how a machine learning model can be moved from experimentation into an **API-based inference service**.

---

## Problem Statement
Customer churn is a critical business problem for telecom companies.  
This project predicts whether a customer is likely to churn based on demographic, service usage, and billing features.

---

## Dataset
- ~7,000 telecom customer records  
- Features include customer demographics, services subscribed, contract details, and billing information

---

## Model & Training
- Algorithm: **Gradient Boosting (XGBoost)**
- Evaluation:
  - F1-score: **0.69**
  - ROC-AUC: **0.88**
- Validation strategy:
  - 5-fold cross-validation
- Hyperparameter tuning:
  - **Optuna**

Model training and experiments were tracked using **MLflow**.

---

## API Design

### Endpoint
`POST /predict`

### Request (JSON)
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "InternetService": "Fiber optic",
  "MonthlyCharges": 85.3,
  "TotalCharges": 1023.6
}
