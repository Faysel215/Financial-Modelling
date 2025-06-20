# Gaussian Process Surrogates for Integrated Option Greek Estimation
## Overview
This project provides a Python implementation of a modern quantitative finance technique for estimating option Greeks (Delta, Gamma) using Gaussian Process (GP) regression. Instead of calculating each Greek with a separate model or formula, this approach first learns a non-parametric surrogate model of the option pricing surface itself. The Greeks are then derived by analytically or numerically differentiating this single, consistent surface.
This method, grounded in recent academic research, offers several key advantages:
- **Mathematical Consistency:** By deriving all Greeks from a single price function, the resulting sensitivities (Delta, Gamma, etc.) are inherently consistent with each other.
- **Uncertainty Quantification:** As a Bayesian method, Gaussian Processes provide not only a point estimate for prices and Greeks but also a state-dependent confidence interval, quantifying the model's own uncertainty.
- **Model-Free Approach:** The GP learns the pricing function directly from data, making fewer restrictive assumptions than traditional parametric models like Black-Scholes-Merton.
- **Handles Noisy Data:** The framework naturally accommodates noisy input data, such as actual market prices or prices from Monte Carlo simulations.
This implementation uses the Black-Scholes-Merton (BSM) model as the "ground truth" to generate synthetic option data. It then trains a GP to learn this pricing function from a noisy sample and compares the GP-derived Greeks against the true analytical Greeks from the BSM model.

## How It Works
The script `gp_surrogate_greeks.py` follows these steps:
1. **Data Generation:**
  - A set of "true" European call option prices is generated using the analytical Black-Scholes-Merton formula across a range of underlying asset prices (`S`).
  - To simulate real-world conditions like market friction or model misspecification, a small amount of random noise is added to these true prices. This noisy dataset serves as the training data.
2. **GP Model Training:**
  - A GaussianProcessRegressor from scikit-learn is trained on the noisy price data.
  - The kernel chosen is a `Matern(nu=2.5)`. This is a critical choice, as a kernel with $ν=2.5$ produces a surrogate model that is twice-differentiable, which is the minimum smoothness required to estimate both Delta (1st derivative) and Gamma (2nd derivative).
  - The kernel also includes a WhiteKernel component to explicitly model the noise present in the training data.
3. **Integrated Greek Estimation:**
- With the GP surrogate for the price surface learned, the Greeks are estimated via numerical differentiation of the GP's mean prediction function:
  - **Delta ($\Delta$)** is calculated using a first-order central difference formula.
  - **Gamma ($\Gamma$)** is calculated using a second-order central difference formula.
4. **Evaluation and Visualization:**
- The script generates a series of plots to visually compare the performance of the GP surrogate against the true BSM model:
  - **Plot 1: Price Approximation:** Shows the noisy training points, the true BSM price curve, the GP's predicted price curve, and the 95% confidence interval.
  - **Plot 2: Delta Estimation:** Compares the GP-derived Delta to the analytical BSM Delta.
  - **Plot 3: Gamma Estimation:** Compares the GP-derived Gamma to the analytical BSM Gamma.

## Requirements
This project requires the following Python libraries:
- numpy
- matplotlib
- scikit-learn
- scipy
You can install them using pip:
```
pip install numpy matplotlib scikit-learn scipy
```

### Usage
To run the simulation and generate the plots, simply execute the Python script:
```
python gp_surrogate_greeks.py
```

### Example Output
Running the script will produce and display a plot with three subplots, demonstrating the effectiveness of the GP surrogate model.

![Greek-plots](estim-greek-plot.png)

The top plot shows how well the GP model (blue dashed line) learns the true price function (black solid line) from the scattered, noisy training data (red 'x' markers). The shaded blue region represents the model's uncertainty.
The middle and bottom plots show that the Delta and Gamma derived from the GP model are remarkably accurate, closely matching the true analytical Greeks.
