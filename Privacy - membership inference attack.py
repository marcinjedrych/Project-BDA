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
import os

# Import your synthetic data generators
from SyntheticGAN import generate_synthetic_data as generate_ctgan
from SyntheticVAE import generate_synthetic_data as generate_vae  # Assuming a similar interface

# Load original data
filepath = os.path.dirname(os.path.realpath(__file__))
inputdatadir = os.path.join(filepath, "Data", "original_train_data.xlsx")

original = pd.read_excel(inputdatadir)

def encode_categorical(data):
    data = pd.get_dummies(data, columns=['stage'], drop_first=True) 
    data['therapy'] = data['therapy'].astype(int)
    data['stage_II'] = data["stage_II"].astype(int)
    data['stage_III'] = data["stage_III"].astype(int)
    data['stage_IV'] = data["stage_IV"].astype(int)
    
    return data


def generate_synthetic(train_original, syn_type):
    """Generate synthetic data using the selected method."""
    if syn_type == 'CTGAN':
        return generate_ctgan(train_original)
    elif syn_type == 'VAE':
        print('VAE function not implemented yet')
        return generate_vae(train_original)
    else:
        raise ValueError("Invalid syn_type. Choose 'CTGAN' or 'VAE'.")

def evaluate_mia(original, syn, test):
    """Can a model tell if a sample is real (from holdout) or synthetic?"""
    
    # Encode the datasets
    original_encoded = encode_categorical(original)
    # synGAN_encoded = encode_categorical(synGAN)
    test_encoded = encode_categorical(test)
    
    # Combine the original and synthetic data for training
    X_train = pd.concat([original_encoded, syn], axis=0, ignore_index=True)
    y_train = np.array([0]*len(original_encoded) + [1]*len(syn))
    
    # Train the classifier on the combined dataset
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    # Test on the holdout set (test data)
    y_test = np.array([0] * len(test_encoded))  # All true labels are 0 (real data)
    
    # Predict on the holdout set
    y_pred = clf.predict(test_encoded)

    # Evaluate the accuracy of the attack model on the test set
    return accuracy_score(y_test, y_pred)

def evaluate_targeted_mia(original, syn, test, n_samples=100):
    """Can an attacker determine whether specific records (individuals) were in the training set?"""
    
    sensitive = original.sample(n=n_samples, random_state=1)
    non_sensitive = test.sample(n=n_samples, random_state=1)
    test_records = pd.concat([sensitive, non_sensitive])
    true_labels = np.array([1]*n_samples + [0]*n_samples)

    scaler = StandardScaler()
    test_encoded = encode_categorical(test_records)

    # Normalize
    syn_normalized = scaler.fit_transform(syn)
    test_normalized = scaler.transform(test_encoded)

    # Compute minimum distances
    min_distances = [
        np.min(distance.cdist([record], syn_normalized, 'euclidean'))
        for record in test_normalized
    ]

    # Median of non-sensitive distances as threshold
    threshold = np.median(min_distances[n_samples:])
    predictions = (np.array(min_distances) < threshold).astype(int)

    return f1_score(true_labels, predictions)


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


mia_score = evaluate_mia(original, synGAN, test)

print(f"Membership Inference Attack Accuracy (CTGAN): {mia_score:.3f}")

targeted_f1 = evaluate_targeted_mia(original, synGAN, test)
print(f"Targeted MIA F1 Score (CTGAN): {targeted_f1:.3f}")


mia_score = evaluate_mia(original, synVAE, test)
print(f"Membership Inference Attack Accuracy (VAE): {mia_score:.3f}")

targeted_f1 = evaluate_targeted_mia(original, synVAE, test)
print(f"Targeted MIA F1 Score (VAE): {targeted_f1:.3f}")
