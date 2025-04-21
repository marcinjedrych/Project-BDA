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
import os

filepath = os.path.dirname(os.path.realpath(__file__))
originaldir = os.path.join(filepath, "Data", "original_train_data.xlsx")
syngandatadir = os.path.join(filepath, "Data", "synthetic_GAN_data.xlsx")
synvaedir = os.path.join(filepath, "Data", "synthetic_VAE_data.xlsx")

original = pd.read_excel(originaldir)
synGAN = pd.read_excel(syngandatadir)  #GAN
synVAE = pd.read_excel(synvaedir)

### --- UNIVARIATE ---

def plot_univariate(original, synthetic1, synthetic2=None):
    print('\nUNIVARIATE PLOTS')
    
    numeric_vars = original.select_dtypes(include=[np.number]).columns
    # Compare numerical distributions
    for var in numeric_vars:
        fig, axes = plt.subplots(1, 2 if synthetic2 is None else 3, figsize=(18, 5), sharey=True)
        sns.histplot(original[var], kde=True, ax=axes[0], color='blue', bins=30)
        sns.histplot(synthetic1[var], kde=True, ax=axes[1], color='red', bins=30)
        
        if synthetic2 is not None:
            sns.histplot(synthetic2[var], kde=True, ax=axes[2], color='green', bins=30)
            axes[2].set_title(f"Synthetic VAE: {var} Distribution")
        
        axes[0].set_title(f"Original: {var} Distribution")
        axes[1].set_title(f"Synthetic GAN: {var} Distribution")
        plt.show()
    
    # Compare categorical distributions
    stage_order = ["I", "II", "III", "IV"]

    # Plot for 'stage'
    fig, axes = plt.subplots(1, 2 if synthetic2 is None else 3, figsize=(18, 5), sharey=True)
    sns.countplot(x=original["stage"], ax=axes[0], order=stage_order)
    sns.countplot(x=synthetic1["stage"], ax=axes[1], order=stage_order)
    
    if synthetic2 is not None:
        sns.countplot(x=synthetic2["stage"], ax=axes[2], order=stage_order)
        axes[2].set_title("Synthetic VAE: Stage Counts")
    
    axes[0].set_title("Original: Stage Counts")
    axes[1].set_title("Synthetic GAN: Stage Counts")
    plt.show()
    
    # Plot for 'therapy'
    fig, axes = plt.subplots(1, 2 if synthetic2 is None else 3, figsize=(18, 5), sharey=True)
    sns.countplot(x=original["therapy"], ax=axes[0])
    sns.countplot(x=synthetic1["therapy"], ax=axes[1])
    
    if synthetic2 is not None:
        sns.countplot(x=synthetic2["therapy"], ax=axes[2])
        axes[2].set_title("Synthetic VAE: Therapy Counts")
    
    axes[0].set_title("Original: Therapy Counts")
    axes[1].set_title("Synthetic GAN: Therapy Counts")
    plt.show()

plot_univariate(original, synGAN, synVAE)

### --- BIVARIATE ---

def plot_bivariate(original, synthetic1, synthetic2=None):
    print('\nBIVARIATE PLOTS')
    
    sns.set(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2 if synthetic2 is None else 3, figsize=(18, 5), sharey=True)
    sns.regplot(x='weight', y='bp', data=original, lowess=True, scatter_kws={'alpha':0.5}, ax=axes[0])
    sns.regplot(x='weight', y='bp', data=synthetic1, lowess=True, scatter_kws={'alpha':0.5}, ax=axes[1])
    
    if synthetic2 is not None:
        sns.regplot(x='weight', y='bp', data=synthetic2, lowess=True, scatter_kws={'alpha':0.5}, ax=axes[2])
        axes[2].set_title("Synthetic VAE: Effect of Weight on Blood Pressure")
    
    axes[0].set_title("Original: Effect of Weight on Blood Pressure")
    axes[1].set_title("Synthetic GAN: Effect of Weight on Blood Pressure")
    plt.show()
    
    fig, axes = plt.subplots(2, 3 if synthetic2 is not None else 2, figsize=(18, 10), sharey=True)
    
    sns.barplot(x='stage', y='bp', data=original, order=['I', 'II', 'III', 'IV'], ax=axes[0, 0])
    sns.barplot(x='stage', y='bp', data=synthetic1, order=['I', 'II', 'III', 'IV'], ax=axes[0, 1])
    
    if synthetic2 is not None:
        sns.barplot(x='stage', y='bp', data=synthetic2, order=['I', 'II', 'III', 'IV'], ax=axes[0, 2])
        axes[0, 2].set_title("Synthetic VAE: Effect of Disease Stage on BP")
    
    axes[0, 0].set_title("Original: Effect of Disease Stage on BP")
    axes[0, 1].set_title("Synthetic GAN: Effect of Disease Stage on BP")
    
    sns.barplot(x='therapy', y='bp', data=original, ax=axes[1, 0])
    sns.barplot(x='therapy', y='bp', data=synthetic1, ax=axes[1, 1])
    
    if synthetic2 is not None:
        sns.barplot(x='therapy', y='bp', data=synthetic2, ax=axes[1, 2])
        axes[1, 2].set_title("Synthetic VAE: Effect of Therapy on BP")
    
    axes[1, 0].set_title("Original: Effect of Therapy on BP")
    axes[1, 1].set_title("Synthetic GAN: Effect of Therapy on BP")
    
    plt.tight_layout()
    plt.show()

plot_bivariate(original, synGAN, synVAE)

def compare_correlation_matrices(original, synthetic1, synthetic2=None):
    original.stage = pd.factorize(original.stage)[0]
    synthetic1.stage = pd.factorize(synthetic1.stage)[0]
    synthetic2.stage = pd.factorize(synthetic2.stage)[0]

    corr_original = original.corr()
    corr_synthetic1 = synthetic1.corr()
    diff_corr_1 = corr_original - corr_synthetic1
    
    if synthetic2 is not None:
        corr_synthetic2 = synthetic2.corr()
        diff_corr_2 = corr_original - corr_synthetic2
    
    fig, axes = plt.subplots(1, 5 if synthetic2 is not None else 3, figsize=(24, 6))

    # Heatmap for original data
    sns.heatmap(corr_original, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Original Data Correlation Matrix")
    
    # Heatmap for synthetic GAN data
    sns.heatmap(corr_synthetic1, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Synthetic GAN Correlation Matrix")
    
    if synthetic2 is not None:
        # Heatmap for synthetic VAE data
        sns.heatmap(corr_synthetic2, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[2])
        axes[2].set_title("Synthetic VAE Correlation Matrix")
        
        # Difference in correlation (diff will range from -2 to +2) for GAN
        sns.heatmap(diff_corr_1, annot=True, cmap='coolwarm', center=0, ax=axes[3])
        axes[3].set_title("Difference in Correlation (Original - GAN)")

        # Difference in correlation (diff will range from -2 to +2) for VAE
        sns.heatmap(diff_corr_2, annot=True, cmap='coolwarm', center=0, ax=axes[4])
        axes[4].set_title("Difference in Correlation (Original - VAE)")
    else:
        # Difference in correlation (diff will range from -2 to +2) for GAN
        sns.heatmap(diff_corr_1, annot=True, cmap='coolwarm', center=0, ax=axes[2])
        axes[2].set_title("Difference in Correlation (Original - GAN)")


    plt.tight_layout()
    plt.show()

compare_correlation_matrices(original, synGAN, synVAE)
