# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:18:11 2025

@author: Marcin
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

original = pd.read_excel("Data/example_data.xlsx" )
synthetic = pd.read_excel("Data/synthetic_data_GAN.xlsx" )

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
 
plot_univariate(original, synthetic)
plot_bivariate(original, synthetic)