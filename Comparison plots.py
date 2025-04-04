# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:18:11 2025

Script to compare original data with synthetic data

@author: Marcin
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

original = pd.read_excel("Data/example_data.xlsx" )
syntheticGAN = pd.read_excel("Data/synthetic_data_GAN.xlsx" )  #GAN
#synthticVAE =  pd.read_excel("Data/synthetic_data_VAE.xlsx" ) 

### --- UNIVARIATE ---

def plot_univariate(original, synthetic):
    
    numeric_vars = original.select_dtypes(include=[np.number]).columns
    categorical_vars = original.select_dtypes(exclude=[np.number]).columns
    
    # Compare numerical distributions
    for var in numeric_vars:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        sns.histplot(original[var], kde=True, ax=axes[0], color='blue', bins=30)
        sns.histplot(synthetic[var], kde=True, ax=axes[1], color='red', bins=30)
        axes[0].set_title(f"Original: {var} Distribution")
        axes[1].set_title(f"Synthetic: {var} Distribution")
        plt.show()
    
    # Compare categorical distributions
    for var in categorical_vars:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        sns.countplot(x=original[var], ax=axes[0])
        sns.countplot(x=synthetic[var], ax=axes[1])
        axes[0].set_title(f"Original: {var} Counts")
        axes[1].set_title(f"Synthetic: {var} Counts")
        plt.show()
 
plot_univariate(original, syntheticGAN)

from sklearn.metrics.pairwise import pairwise_kernels
def maximum_mean_discrepancy(original, synthetic, kernel='rbf'):
    numeric_vars = original.select_dtypes(include=[np.number]).columns
    mmd_values = {}
    for var in numeric_vars:
        x = original[var].dropna().values.reshape(-1, 1)
        y = synthetic[var].dropna().values.reshape(-1, 1)
        K_xx = pairwise_kernels(x, x, metric=kernel).mean()
        K_yy = pairwise_kernels(y, y, metric=kernel).mean()
        K_xy = pairwise_kernels(x, y, metric=kernel).mean()
        mmd = K_xx + K_yy - 2 * K_xy
        mmd_values[var] = mmd
    
    # Plot MMD values
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(mmd_values.keys()), y=list(mmd_values.values()), color='green')
    plt.xticks(rotation=45)
    plt.title("Maximum Mean Discrepancy (MMD) between Original and Synthetic Data")
    plt.ylabel("MMD Value")
    plt.show()
    
    return mmd_values

maximum_mean_discrepancy(original, syntheticGAN)

### --- BIVARIATE ---

def plot_bivariate(original, synthetic):
    sns.set(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.violinplot(x='stage', y='age', data=original, order=['I', 'II', 'III', 'IV'], ax=axes[0])
    sns.violinplot(x='stage', y='age', data=synthetic, order=['I', 'II', 'III', 'IV'], ax=axes[1])
    axes[0].set_title("Original: Effect of Age on Disease Stage")
    axes[1].set_title("Synthetic: Effect of Age on Disease Stage")
    plt.show()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.regplot(x='weight', y='bp', data=original, lowess=True, scatter_kws={'alpha':0.5}, ax=axes[0])
    sns.regplot(x='weight', y='bp', data=synthetic, lowess=True, scatter_kws={'alpha':0.5}, ax=axes[1])
    axes[0].set_title("Original: Effect of Weight on Blood Pressure")
    axes[1].set_title("Synthetic: Effect of Weight on Blood Pressure")
    plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    sns.barplot(x='stage', y='bp', data=original, order=['I', 'II', 'III', 'IV'], ax=axes[0, 0])
    sns.barplot(x='stage', y='bp', data=synthetic, order=['I', 'II', 'III', 'IV'], ax=axes[0, 1])
    axes[0, 0].set_title("Original: Effect of Disease Stage on BP")
    axes[0, 1].set_title("Synthetic: Effect of Disease Stage on BP")
    sns.barplot(x='therapy', y='bp', data=original, ax=axes[1, 0])
    sns.barplot(x='therapy', y='bp', data=synthetic, ax=axes[1, 1])
    axes[1, 0].set_title("Original: Effect of Therapy on BP")
    axes[1, 1].set_title("Synthetic: Effect of Therapy on BP")
    plt.show()


plot_bivariate(original, syntheticGAN)

def compare_correlation_matrices(original, synthetic):
    corr_original = original.corr()
    corr_synthetic = synthetic.corr()
    diff_corr = corr_original - corr_synthetic
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.heatmap(corr_original, annot=True, cmap='coolwarm', ax=axes[0])
    axes[0].set_title("Original Data Correlation Matrix")
    sns.heatmap(corr_synthetic, annot=True, cmap='coolwarm', ax=axes[1])
    axes[1].set_title("Synthetic Data Correlation Matrix")
    sns.heatmap(diff_corr, annot=True, cmap='coolwarm', ax=axes[2])
    axes[2].set_title("Difference in Correlation Matrices")
    plt.show()


compare_correlation_matrices(original, syntheticGAN)