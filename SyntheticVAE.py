import sdv
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
import pandas as pd
import os
import matplotlib.pyplot as plt
from sdv.evaluation.single_table import evaluate_quality

def generate_synthetic_data(df, epochs = 500, visualize_loss = False, evaluation = False):
    metadata = Metadata.detect_from_dataframe(data = df)
    metadata.visualize()

    synthesizer = TVAESynthesizer(
        metadata,   
        epochs=epochs
    )
    synthesizer.fit(df)
    
    if visualize_loss:
        loss = synthesizer.get_loss_values()

        loss_end_each_epoch = loss[loss["Batch"]==15]
        plt.subplot()
        plt.plot(loss_end_each_epoch["Epoch"], loss_end_each_epoch["Loss"], label='Loss')
        plt.legend(loc='upper right')
        plt.title('Loss')
        plt.show()

    if evaluation:
        evaluate_quality(
            real_data=df,
            synthetic_data=synthetic,
            metadata=metadata
            )

    synthetic_data = synthesizer.sample(num_rows=len(df))
    
    return synthetic_data

if __name__ == "__main__": 
    filepath = os.path.dirname(os.path.realpath(__file__))
    inputdatadir = os.path.join(filepath, "Data", "original_train_data.xlsx")
    outputdatadir = os.path.join(filepath, "Data", "synthetic_VAE_data.xlsx")

    df = pd.read_excel(inputdatadir)
    synthetic = generate_synthetic_data(df, epochs = 500, visualize_loss = True, evaluation = True)
    synthetic.to_excel(outputdatadir, index=False)

    generate_synthetic_data(df, epochs = 50, visualize_loss=True, evaluation = True)


