# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:14:54 2025

Code to evaluate predictive performance on original vs. synthetic GAN vs. synthetic VAE

@author: Marcin
"""
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import os


# Function to handle categorical variables
def encode_categorical(data):
    
    data = pd.get_dummies(data, columns=['stage'], drop_first=True) 
    data['therapy'] = data['therapy'].astype(int)
    data['stage_II'] = data["stage_II"].astype(int)
    data['stage_III'] = data["stage_III"].astype(int)
    data['stage_IV'] = data["stage_IV"].astype(int)

    
    return data

# Function to perform the regression model
def linear_regression(data, target_variable, test_data):
    
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




def compare_models(modelorig, modelgan, modelvae):
    
    # Extract coefficients from each model
    coeffs_orig = modelorig.params
    coeffs_GAN = modelgan.params
    coeffs_VAE = modelvae.params
    
    # Create a DataFrame to compare coefficients
    coefficients_comparison = pd.DataFrame({
        'Coeff Model 1 (Original Data)': coeffs_orig,
        'Coeff Model 2 (Synthetic GAN Data)': coeffs_GAN,
        'Coeff Model 3 (Synthetic VAE Data)': coeffs_VAE,
        'Difference Model 2 - Model 1': coeffs_GAN - coeffs_orig,
        'Difference Model 3 - Model 1': coeffs_VAE - coeffs_orig
    })
    
    # Print the model summaries
    print(coefficients_comparison)

    coefficients_comparison.to_csv(os.path.join(filepath, "Data", "comparison_coefficients.csv")
)
    
    # Return the table of coefficients and differences
    return coefficients_comparison

if __name__ == "__main__": 

    # Load the data
    filepath = os.path.dirname(os.path.realpath(__file__))
    originaldir = os.path.join(filepath, "Data", "original_train_data.xlsx")
    syngandatadir = os.path.join(filepath, "Data", "synthetic_GAN_data.xlsx")
    synvaedir = os.path.join(filepath, "Data", "synthetic_VAE_data.xlsx")
    testdir = os.path.join(filepath, "Data", "test_data.xlsx")

    original = pd.read_excel(originaldir)
    synGAN = pd.read_excel(syngandatadir)  #GAN
    synVAE = pd.read_excel(synvaedir)
    test = pd.read_excel(testdir)

    # Run the model for original data
    print("Model trained on ORIGINAL data:")
    original = encode_categorical(original)
    original_model = linear_regression(original, 'bp', test)

    # Run the model for synthetic GAN data
    print("\nModel trained on SYNTHETIC GAN data:")
    synGAN_model = linear_regression(synGAN, 'bp', test)

    # Run the model for synthetic VAE data
    print("\nModel trained on SYNTHETIC VAE data:")
    synVAE_model = linear_regression(synVAE, 'bp', test)

    coefficients_comparison = compare_models(original_model, synGAN_model, synVAE_model)