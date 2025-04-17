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

original = pd.read_excel("Data/original_train_data.xlsx" )
synGAN = pd.read_excel("Data/synthetic_GAN_data.xlsx" )  #GAN
#synVAE =  pd.read_excel("Data/synthetic_VAE_data.xlsx" ) 

### --- UNIVARIATE ---

def plot_univariate(original, synthetic):
    
    print('\nUNIVARIATE PLOTS')
    
    numeric_vars = original.select_dtypes(include=[np.number]).columns
    # Compare numerical distributions
    for var in numeric_vars:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        sns.histplot(original[var], kde=True, ax=axes[0], color='blue', bins=30)
        sns.histplot(synthetic[var], kde=True, ax=axes[1], color='red', bins=30)
        axes[0].set_title(f"Original: {var} Distribution")
        axes[1].set_title(f"Synthetic: {var} Distribution")
        plt.show()
    
    # Compare categorical distributions
    stage_order = ["I", "II", "III", "IV"]

    # Plot for 'stage'
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.countplot(x=original["stage"], ax=axes[0], order=stage_order)
    sns.countplot(x=synthetic["stage"], ax=axes[1], order=stage_order)
    axes[0].set_title("Original: Stage Counts")
    axes[1].set_title("Synthetic: Stage Counts")
    plt.show()
    
    # Plot for 'therapy'
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.countplot(x=original["therapy"], ax=axes[0])
    sns.countplot(x=synthetic["therapy"], ax=axes[1])
    axes[0].set_title("Original: Therapy Counts")
    axes[1].set_title("Synthetic: Therapy Counts")
    plt.show()
 
plot_univariate(original, synGAN)

### --- BIVARIATE ---

def plot_bivariate(original, synthetic):
    
    print('\nBIVARIATE PLOTS')
    
    sns.set(style="whitegrid")
    
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
    plt.tight_layout()
    plt.show()


plot_bivariate(original, synGAN)

def compare_correlation_matrices(original, synthetic):
    corr_original = original.corr()
    corr_synthetic = synthetic.corr()
    diff_corr = corr_original - corr_synthetic
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Heatmap for original data
    sns.heatmap(corr_original, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Original Data Correlation Matrix")
    
    # Heatmap for synthetic data
    sns.heatmap(corr_synthetic, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Synthetic GAN Correlation Matrix")
    
    # Difference in correlation (diff will range from -2 to +2)
    sns.heatmap(diff_corr, annot=True, cmap='coolwarm', center=0, ax=axes[2])
    axes[2].set_title("Difference in Correlation Matrices")
    
    plt.tight_layout()
    plt.show()


compare_correlation_matrices(original, synGAN)