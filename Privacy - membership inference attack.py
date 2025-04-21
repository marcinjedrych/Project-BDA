# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:14:54 2025

 Membership Inference Attack (MIA): How well an attacker can determine if a specific individual’s data was used to train the synthetic data generator.
 
1.) Split your original data into a training set (used to generate synthetic data) and a holdout set (never seen by the synthesizer).
2.) Train a classifier to distinguish between: Samples from the original dataset and samples from the synthetic dataset.
3.) Measure the classifier’s accuracy on the holdout set. If it performs poorly (≈50% for binary classification), the synthetic data preserves privacy.

MIA Accuracy > 65%: High privacy risk (synthetic data leaks information about the training set).

@author: Marcin
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import os

# Import your synthetic data generators
from SyntheticGAN import generate_synthetic_data as generate_ctgan
from SyntheticVAE import generate_synthetic_data as generate_vae  # Assuming a similar interface

# Load original data
filepath = os.path.dirname(os.path.realpath(__file__))
inputdatadir = os.path.join(filepath, "Data", "original_train_data.xlsx")

original = pd.read_excel(inputdatadir)

def encode_categorical(df):
    """One-hot encode categorical columns."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    return pd.get_dummies(df, columns=categorical_cols)

def generate_synthetic(train_original, syn_type):
    """Generate synthetic data using the selected method."""
    if syn_type == 'CTGAN':
        return generate_ctgan(train_original)
    elif syn_type == 'VAE':
        print('VAE function not implemented yet')
        return generate_vae(train_original)
    else:
        raise ValueError("Invalid syn_type. Choose 'CTGAN' or 'VAE'.")

def evaluate_mia(original, syn_type):
    
    """	Can a model tell in general if a sample is real (from holdout) or synthetic?, 
    = overall leakage """
    
    train_original, holdout_original = train_test_split(original, test_size=0.3, random_state=42)

    synthetic = generate_synthetic(train_original, syn_type)
    
    # Encode data
    holdout_encoded = encode_categorical(holdout_original)
    synthetic_encoded = encode_categorical(synthetic)

    # Combine datasets and label
    X = pd.concat([holdout_encoded, synthetic_encoded], axis=0, ignore_index=True)
    y = np.array([0]*len(holdout_encoded) + [1]*len(synthetic_encoded))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def evaluate_targeted_mia(original, syn_type, n_samples=100):
    
    """Can an attacker determine whether specific records (e.g., individuals) were in the training set?
        = individual level privacy risk
    """
    train_original, holdout_original = train_test_split(original, test_size=0.3, random_state=42)
    synthetic = generate_synthetic(train_original, syn_type)

    sensitive = train_original.sample(n=n_samples, random_state=1)
    non_sensitive = holdout_original.sample(n=n_samples, random_state=1)
    test_records = pd.concat([sensitive, non_sensitive])
    true_labels = np.array([1]*n_samples + [0]*n_samples)

    scaler = StandardScaler()
    synthetic_encoded = encode_categorical(synthetic)
    test_encoded = encode_categorical(test_records)

    synthetic_normalized = scaler.fit_transform(synthetic_encoded)
    test_normalized = scaler.transform(test_encoded)

    # Compute minimum distances
    min_distances = [
        np.min(distance.cdist([record], synthetic_normalized, 'euclidean'))
        for record in test_normalized
    ]

    # Median of non-sensitive distances as threshold
    threshold = np.median(min_distances[n_samples:])
    predictions = (np.array(min_distances) < threshold).astype(int)

    return f1_score(true_labels, predictions)

# --- Example usage --- #

mia_score = evaluate_mia(original, syn_type='CTGAN')
print(f"Membership Inference Attack Accuracy (CTGAN): {mia_score:.3f}")

targeted_f1 = evaluate_targeted_mia(original, syn_type='CTGAN')
print(f"Targeted MIA F1 Score (CTGAN): {targeted_f1:.3f}")


mia_score = evaluate_mia(original, syn_type='VAE')
print(f"Membership Inference Attack Accuracy (VAE): {mia_score:.3f}")

targeted_f1 = evaluate_targeted_mia(original, syn_type='VAE')
print(f"Targeted MIA F1 Score (VAE): {targeted_f1:.3f}")
