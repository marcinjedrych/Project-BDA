# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:14:54 2025

Code to evaluate predictive performance on original vs. synthetic GAN vs. synthetic VAE

@author: Marcin
"""
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
original = pd.read_excel("Data/original_train_data.xlsx")
synGAN = pd.read_excel("Data/synthetic_GAN_data.xlsx")
#synVAE = .....
test = pd.read_excel("Data/test_data.xlsx")

# Function to handle categorical variables
def encode_categorical(data):

    data = pd.get_dummies(data, columns=['stage'], drop_first=True) 
    data['therapy'] = data['therapy'].astype(int)
    
    return data

# Function to perform the regression model
def linear_regression(data, target_variable, test_data):
    
    data = encode_categorical(data)
    test_data = encode_categorical(test_data)
    
    # Separate target and predictors
    X_train = data.drop(columns=[target_variable])
    y_train = data[target_variable]
    
    X_test = test_data.drop(columns=[target_variable])
    y_test = test_data[target_variable]
    
    # Add constant (intercept) to the predictors
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    
    # Fit the OLS model
    model = sm.OLS(y_train, X_train).fit()

    # Predictions on test set
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1] - 1  # Exclude intercept
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Print the model summary and coefficients
    print(model.summary())
    
    # Print evaluation metrics
    print('\nOn Test data:')
    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Adjusted R-squared: {adj_r2:.4f}\n")
    
    return model

# Run the model for original data
print("Model trained on ORIGINAL data:")
original_model = linear_regression(original, 'bp', test)

# Run the model for synthetic GAN data
print("\nModel trained on SYNTHETIC GAN data:")
synGAN_model = linear_regression(synGAN, 'bp', test)

# Run the model for synthetic VAE data
# print("\nModel trained on SYNTHETIC VAE data:")
# synVAE_model = linear_regression(synVAE, 'bp', test)


def compare_models(model1, model2):
    
    # Extract coefficients from each model
    coeffs_model1 = model1.params
    coeffs_model2 = model2.params
    
    # Create a DataFrame to compare coefficients
    coefficients_comparison = pd.DataFrame({
        'Coeff Model 1 (Original Data)': coeffs_model1,
        'Coeff Model 2 (Synthetic GAN Data)': coeffs_model2,
        'Difference': coeffs_model2 - coeffs_model1
    })
    
    # Print the model summaries
    print(coefficients_comparison)
    
    # Return the table of coefficients and differences
    return coefficients_comparison

coefficients_comparison = compare_models(original_model, synGAN_model)