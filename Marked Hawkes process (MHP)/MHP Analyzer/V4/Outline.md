This version elevates the application from an exploratory tool to a comparative analytical platform. The key upgrades are:
Dual Estimator Framework: The application is no longer limited to a single non-parametric model. Users can now select and directly compare a Linear (Parametric) model against the Neural Network (Non-Parametric) model.
Quantitative Model Comparison: To support the dual estimator framework, the app now calculates and displays key statistical metrics: Maximum Log-Likelihood, AIC (Akaike Information Criterion), and BIC (Bayesian Information Criterion). This allows for a rigorous, data-driven assessment of which model provides a better fit while penalizing for complexity.
Deep Model Explanation with SHAP: For the neural network model, we've integrated the shap library to provide deep, theoretically-grounded explanations. This moves beyond visual interpretation to quantify feature importance and interactions.
Streamlined UI: The user interface has been updated with a model selector dropdown to manage the increased functionality in a clean and intuitive way.
1. Introduction: A Comparative Analysis Tool
1.1. Core Purpose: To quantitatively compare how well parametric (linear) and non-parametric (neural network) models can explain the dynamics of marked point processes.
1.2. Key Analytical Questions Answered:
Is the relationship between event characteristics (marks) and future volatility linear or non-linear?
Which model provides a better statistical fit to the data once model complexity is accounted for (via AIC/BIC)?
What are the most important drivers of influence according to the more complex neural network model?
2. The Dual Modeling Workflow
2.1. Foundation (Shared Steps)
Data Synthesis & Event Detection: Unchanged. Events are identified from a synthetic volatility series based on a user-defined threshold.
Mark Assignment & Normalization: Unchanged. Events are marked with sentiment/odds data, which is then normalized.
2.2. Path A: Linear (Parametric) Estimator
Influence Function κ(m): Assumes a simple linear form: κ(m) = β₁ * sentiment + β₂ * odds_change.
Model Fitting: Uses scipy.optimize.minimize to find the μ, α, β₁, and β₂ parameters that maximize the log-likelihood function (neg_log_likelihood_parametric).
Output: A set of four easily interpretable coefficients.
2.3. Path B: Neural Network (Non-Parametric) Estimator
Influence Function κ(m): Assumes no specific form. It is approximated by a Multi-Layer Perceptron (MLP).
Model Fitting (Two-Stage):
MLP Training: The neural network is first trained to predict a proxy for influence (the number of subsequent events in a window).
Hawkes Parameter Fitting: The trained MLP is then treated as a fixed component inside the Hawkes log-likelihood function, and scipy.optimize.minimize is used to find the optimal μ and α.
Output: A trained neural network object, which is then explained using SHAP.
3. Application UI & Enhanced Interactivity
3.1. Sidebar Controls
Estimator Type Selectbox (New): The primary control for switching between the "Linear (Parametric)" and "Neural Network (Non-Parametric)" analysis paths.
Contextual Controls: The sidebar now dynamically shows/hides the "Number of Hidden Neurons" slider depending on whether the Neural Network model is selected.
3.2. Main Panel Displays
Section 1: Event Data: Displays the event data table and time series plots.
Section 2: Model Fitting & Explanation: This section is now dynamic.
If Linear is selected, it displays the four estimated coefficients (μ, α, β₁, β₂) in metric boxes.
If Neural Network is selected, it displays the estimated μ and α, followed by the SHAP Summary Plot for feature importance.
Section 3: Model Comparison (New): A dedicated section at the bottom displays the Log-Likelihood, AIC, and BIC for the currently selected and fitted model, with tooltips explaining how to interpret the scores.
4. Advanced Interpretation & Analysis
4.1. For the Linear Model:
Direct Interpretation: The magnitude and sign of the β₁ and β₂ coefficients provide a straightforward measure of each feature's influence.
4.2. For the Neural Network Model:
SHAP Summary Plot: This is now the primary tool for explanation.
Feature Importance: Features are ranked by the mean absolute SHAP value.
Impact Direction: The color coding (red for high feature value, blue for low) shows how each feature's value affects the model's output (the predicted influence κ(m)).
4.3. The Core Analytical Task: Comparing the Models
The User's Goal: To select the Linear model, note its AIC/BIC. Then, select the Neural Network model and note its AIC/BIC.
Conclusion: The model with the lower AIC and BIC is considered the better fit for the data. A significantly lower score for the neural network provides strong evidence that non-linear relationships are present and important.
