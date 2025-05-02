# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:11:54 2025

@author: Marcin
"""

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import os

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

def encode_categorical(data):
    data = pd.get_dummies(data, columns=['stage'], drop_first=True) 
    data['therapy'] = data['therapy'].astype(int)
    data['stage_II'] = data["stage_II"].astype(int)
    data['stage_III'] = data["stage_III"].astype(int)
    data['stage_IV'] = data["stage_IV"].astype(int)

    
    return data

if __name__ == "__main__": 
    # Load data
    filepath = os.path.dirname(os.path.realpath(__file__))
    originaldir = os.path.join(filepath, "Data", "original_train_data.xlsx")

    df = pd.read_excel(originaldir)
    df = encode_categorical(df)

    # Generate synthetic data
    synthetic = generate_synthetic_data(df)

    # Export
    outputdatadir = os.path.join(filepath, "Data", "synthetic_GAN_data.xlsx")
    synthetic.to_excel(outputdatadir, index=False)

    print("Processing complete.")