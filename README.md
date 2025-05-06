# Project-BDA


This project was developed for an assignment in the course Big Data Algorithms. Within this project, a very short introduction into synthetic data was developped. The slides of this introduction can be found in this repository. This code was developed to illustrate how synthetic data can look like and how synthetic data can be evaluated. Some figures produced by this code, can be found in the presentation.

In the code, we first simulated an original dataset and generated synthetic versions using two generative models: **CTGAN** and **TVAE**. We focused on evaluating the synthetic data and the privacy-utility trade-off. 

The methods to generate and evaluate synthetic data are by no means exhaustive. GAN and VAE are two popular generation techniques based on neural networks, hence we chose to variations of these techniques to showcase how synthetic data generation works. As for the evaluation, there is no common way researchers agree on how to properly evaluate similarity, utility or privacy of synthetic data. Therefor, a small selection of what seems to be regularly done is demonstrated here. 


## 📂 Project Structure

Project-BDA/
- Data/                         :  Contains real and synthetic datasets
- `Comparison_plots.py`           :  Univariate and bivariate visual comparisons to evaluate similarity between original and synthetic data.
- `introduction_synthetic_data_generation.pdf`  : Slides of the short introduction to synthetic data generation.
- `Privacy.py`                    :  The evaluation of the Membership Inference Attack measure for privacy.
- `Simulation_example_data.py` : Simulates data for original dataset.
- `SyntheticGAN.py`               : Generates synthetic data using CTGAN.
- `SyntheticVAE.py`               : Generates synthetic data using TVAE.
- `Utility.py`                    : Predictive performance evaluations to compare how useful a synthetic dataset is compared to the original data.



## 🚀 How to run

1. Simulate the original dataset
    - Run: `Simulation_example_data.py`

2. Generate synthetic data
    - Using CTGAN: `SyntheticGAN.py`
    - Using TVAE: `SyntheticVAE.py`

3. Compare datasets visually
    - Run: `Comparison_plots.py`

4. Evaluate utility (predictive performance)
    - Run: `Utility.py`

5. Evaluate privacy (membership inference attack)
   - Run: `Privacy.py`

## 🔧 Requirements

- Python 3.8+
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `sdv` (for CTGAN and TVAE models)

## 👩‍💻 Authors

Developed by **Marcin Jedrych, Frie Van Bauwel and Xueting Li** as part of the course **Big Data Algorithms (BDA)** at Ghent University.  

## :memo: Reference
This project makes use of the sdv-package, which is developed by:
Patki, N., Wedge, R., & Veeramachaneni, K. (2016). The Synthetic Data Vault. 2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA), 399–410. https://doi.org/10.1109/DSAA.2016.49

Sources used to develop the presentation, are listed on the last slide of the presentation.
