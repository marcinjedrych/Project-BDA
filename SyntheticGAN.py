# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:11:54 2025

@author: Marcin
"""

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


# metadata = {
#     "primary_key": None,  # No explicit primary key in the dataset
#     "columns": {
#         "age": {"sdtype": "numerical", "computer_representation": "Float"},
#         "weight": {"sdtype": "numerical", "computer_representation": "Float"},
#         "stage": {"sdtype": "categorical"},
#         "therapy": {"sdtype": "categorical"},
#         "bp": {"sdtype": "numerical", "computer_representation": "Float"},
#     }
# }

def generate_synthetic_data(df):
    metadata_obj = SingleTableMetadata()
    metadata_obj.detect_from_dataframe(df)
    synthesizer = CTGANSynthesizer(metadata_obj)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=len(df))
    
    return synthetic_data


# Load data
df = pd.read_excel("Data/original_train_data.xlsx" )

# Generate synthetic data
synthetic = generate_synthetic_data(df)

# Export 
synthetic.to_excel('Data/synthetic_GAN_data.xlsx', index=False)

print("Processing complete.")