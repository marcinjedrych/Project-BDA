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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

<<<<<<< Updated upstream
# Import your synthetic data generators
from SyntheticGAN import generate_synthetic_data as generate_ctgan
from SyntheticVAE import generate_synthetic_data as generate_vae  # Assuming a similar interface
=======
original = pd.read_excel("Data/original_train_data.xlsx")
synGAN = pd.read_excel("Data/synthetic_GAN_data.xlsx")
test = pd.read_excel("Data/test_data.xlsx")
>>>>>>> Stashed changes

# Load original data
original = pd.read_excel("Data/original_train_data.xlsx")

def encode_categorical(df):
    """One-hot encode categorical columns."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    return pd.get_dummies(df, columns=categorical_cols)


def evaluate_mia(original, synGAN, test):
    """Can a model tell if a sample is real (from holdout) or synthetic?"""
    
    # Encode the datasets
    original_encoded = encode_categorical(original)
    synGAN_encoded = encode_categorical(synGAN)
    test_encoded = encode_categorical(test)
    
    # Combine the original and synthetic data for training
    X_train = pd.concat([original_encoded, synGAN_encoded], axis=0, ignore_index=True)
    y_train = np.array([0]*len(original_encoded) + [1]*len(synGAN_encoded))
    
    # Train the classifier on the combined dataset
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    # Test on the holdout set (test data)
    y_test = np.array([0] * len(test_encoded))  # All true labels are 0 (real data)
    
    # Predict on the holdout set
    y_pred = clf.predict(test_encoded)

    # Evaluate the accuracy of the attack model on the test set
    return accuracy_score(y_test, y_pred)

def evaluate_targeted_mia(original, synGAN, test, n_samples=100):
    """Can an attacker determine whether specific records (individuals) were in the training set?"""
    
    sensitive = original.sample(n=n_samples, random_state=1)
    non_sensitive = test.sample(n=n_samples, random_state=1)
    test_records = pd.concat([sensitive, non_sensitive])
    true_labels = np.array([1]*n_samples + [0]*n_samples)

    scaler = StandardScaler()
    synGAN_encoded = encode_categorical(synGAN)
    test_encoded = encode_categorical(test_records)

    # Normalize
    synGAN_normalized = scaler.fit_transform(synGAN_encoded)
    test_normalized = scaler.transform(test_encoded)

    # Compute minimum distances
    min_distances = [
        np.min(distance.cdist([record], synGAN_normalized, 'euclidean'))
        for record in test_normalized
    ]

    # Median of non-sensitive distances as threshold
    threshold = np.median(min_distances[n_samples:])
    predictions = (np.array(min_distances) < threshold).astype(int)

    return f1_score(true_labels, predictions)


# --- Example usage --- #

mia_score = evaluate_mia(original, synGAN, test)
print(f"Membership Inference Attack Accuracy (CTGAN): {mia_score:.3f}")

targeted_f1 = evaluate_targeted_mia(original, synGAN, test)
print(f"Targeted MIA F1 Score (CTGAN): {targeted_f1:.3f}")
