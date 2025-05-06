# Project-BDA


This project was developed for an assignment in the course Big Data Algorithms. First, we simulated an original dataset and generated synthetic versions using two generative models: **CTGAN** and **TVAE**. We focused on evaluating the synthetic data and the privacy-utility trade-off. 

## 📂 Project Structure

Project-BDA/
- Data/                         :  Contains real and synthetic datasets
- `Comparison_plots.py`           :  Univariate and bivariate comparisons
- `Privacy.py`                    :  Membership inference attack
- `Simulation of example data.py` : Simulates original datasets
- `SyntheticGAN.py`               : Generates synthetic data using CTGAN
- `SyntheticVAE.py`               : Generates synthetic data using TVAE
- `Utility.py`                    : Predictive performance evaluations
- `demo.ipynb`                    : 

## 🚀 How to run

1. Simulate the original dataset
    - Run: `Simulation of example data.py`

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

Developed by Marcin Jedrych, Frie Van Bauwel and Xueting Li as part of the course **Big Data Algorithms (BDA)** at Ghent University.  
