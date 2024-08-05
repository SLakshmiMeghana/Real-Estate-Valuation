# Real Estate Valuation

## Introduction
Real Estate Valuation is a project aimed at predicting the value of properties using various machine learning techniques. This repository contains the code and resources used to preprocess data, train models, and evaluate their performance.

## Features
- Data preprocessing and feature engineering
- Model training using different algorithms
- Model evaluation and comparison
- Hyperparameter tuning
- Visualization of predictions and model performance

## Model
This project uses multiple machine learning models to predict real estate values, including:
- `LinearRegression`: Linear Regression
- `RobustRegression`: HuberRegressor
- `RidgeRegression`: Ridge
- `ElasticNet`: ElasticNet
- `LassoRegression`: Lasso
- `PolynomialRegression`: Polynomial Regression
- `SGDRegressor`: SGDRegressor
- `ANN`: MLPRegressor (Artificial Neural Network) with hidden_layer_sizes=(100,) and max_iter=1000
- `RandomForest`: RandomForestRegressor
- `SVM`: SVR (Support Vector Machine)
- `LGBM`: LGBMRegressor (LightGBM)
- `XGBoost`: XGBRFRegressor (XGBoost)
- `KNN`: KNeighborsRegressor

The models are trained on the preprocessed dataset and their performances are compared.

## Results
The results include various performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.




