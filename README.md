# Loan Approval Prediction

This project is a machine learning application that predicts loan approval status based on various applicant details. The dataset is sourced from Analytics Vidhya.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Setup and Execution](#setup-and-execution)
5. [Data Handling and Preprocessing](#data-handling-and-preprocessing)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Results](#results)
8. [Future Improvements](#future-improvements)

---

## Introduction

The objective of this project is to predict whether a loan application will be approved (`Loan_Status`) using various demographic, financial, and social features of the applicant. 

This is achieved by preprocessing the dataset, handling missing values, encoding categorical data, and training machine learning models.

---

## Features

The dataset includes the following features:
- **Gender**: Male/Female
- **Married**: Applicant's marital status
- **Dependents**: Number of dependents
- **Education**: Applicant's education level
- **Self_Employed**: Employment status of the applicant
- **ApplicantIncome**: Applicant's income
- **CoapplicantIncome**: Co-applicant's income
- **LoanAmount**: Loan amount requested
- **Loan_Amount_Term**: Loan repayment term
- **Credit_History**: Credit history of the applicant
- **Property_Area**: Type of area (Urban/Semiurban/Rural)
- **Loan_Status**: Target variable (Y/N)

---

## Dependencies

The following libraries are required to run this project:
- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- Google Colab (optional)

Install the dependencies using pip:

```bash
pip install pandas numpy scikit-learn xgboost
```

## Setup and Execution

Mount the Google Drive if using Google Colab to run the program and load the dataset into a data frame.

## Data Handling and Preprocessing

**Handling Missing Values**
Different strategies were used to handle missing values based on the nature of the features:
- **Categorical Features**: 
  - Used the mode (most frequent value) to fill missing values in features like `Gender`, `Marital Status`, and `Dependents`.
- **Numerical Features**:
  - Used the mean to fill missing values in features like `LoanAmount` and `Loan_Amount_Term`.

These approaches ensures that the handling methods were appropriate for each feature type, preserving the integrity of the data.

**Label Encoding**
To prepare the data for machine learning models, categorical features were encoded using label encoding. This process converted categories into numeric values that are interpretable by the model, without introducing unnecessary complexity.

This preprocessing ensures the dataset is clean and ready for model training while retaining the meaningful relationships in the data.

## Model Training and Evaluation

Several models were trained and evaluated to predict loan approval status:

### 1. Logistic Regression
- **Description**: A simple linear model for binary classification.
- **Accuracy**: `0.798`.

### 2. XGBoost Classifier (Default Parameters)
- **Description**: Gradient boosting model with default settings.
- **Accuracy**: `0.768`.

### 3. XGBoost Classifier (Tuned Parameters)
- **Description**: XGBoost with hyperparameter tuning:
  - `n_estimators`: 100
  - `learning_rate`: 0.05
  - `max_depth`: 7
  - `subsample`: 0.75 and 0.5
- **Accuracy**:
  - `subsample=0.75`: `0.793`
  - `subsample=0.5`: `0.802`.

## Results

The models showed comparable performance, with minimal differences in accuracy. Below is the summary of results:

| Model                    | Accuracy  |
|--------------------------|-----------|
| Logistic Regression      | `0.798`   |
| XGBoost (Default)        | `0.768`   |
| XGBoost (Tuned, 0.75)    | `0.793`   |
| XGBoost (Tuned, 0.5)     | `0.802`   |

**Observations**
1. Logistic Regression achieved performance similar to XGBoost, demonstrating that simpler models can be effective for this dataset.
2. The XGBoost model with tuned parameters performed best, highlighting its ability to capture complex patterns in the data.

## Future Improvements
- Implement advanced feature engineering techniques.
- Try additional models like Random Forest or Neural Networks.
- Perform hyperparameter tuning using GridSearchCV.
- Use cross-validation for more reliable performance metrics.
