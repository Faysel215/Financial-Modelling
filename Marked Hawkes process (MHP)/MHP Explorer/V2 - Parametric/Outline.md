# The Parametric Model Design

## 1. The Core Equation

Our goal is to model the intensity of volatility spikes, λ(t). We'll use the specific parametric form outlined in the guide. The full intensity equation for a new volatility event at time t will be:

$\lambda(t) = \mu + \sum_{t_i < t} (\beta_1 \cdot \text{sentiment}_i + \beta_2 \cdot \text{odds_change}_i) \cdot \alpha e^{-\alpha(t - t_i)}$

Here's what each part represents in our design:

μ (Baseline Intensity): This will be estimated from the data. It represents the average rate of "unexplained" volatility spikes that happen without being triggered by recent events.
sentiment_i & odds_change_i: These are the values from your feature matrix at the time of a past volatility spike t_i. It's crucial that these are normalized (e.g., scaled to have a mean of 0 and a standard deviation of 1) so we can fairly compare their coefficients.
β₁ & β₂ (The Key Coefficients): These are the numbers we want to find. They will tell us the relative power of each feature in amplifying future volatility.
α e^{-\alpha(t - t_i)} (The Decay Kernel): We're choosing an exponential decay, which is standard. The parameter α controls the "memory" of the process. The model will also estimate the best-fit α from the data.

## 2. Data Preparation and Alignment

This is the most critical practical step in the design:

Step A (Volatility Events): Define a precise rule for an "event". For example: "A VIX event occurs on any day the VIX closes more than 1.5 standard deviations above its 20-day moving average." This gives you your list of timestamps t_1, t_2, ....
Step B (Mark Alignment): For each timestamp t_i, you must find the corresponding sentiment score and Polymarket odds. Since your data might not be recorded at the exact same millisecond, you'll need an alignment rule. A common one is: "For a VIX event at time t_i, use the sentiment score and Polymarket odds from the closest record immediately preceding t_i."
Step C (Normalization): Before feeding the marks into the model, transform them. For each feature (sentiment and odds), subtract its mean and divide by its standard deviation. This ensures a β₁ of 0.5 is directly comparable to a β₂ of 0.2.

## 3. Hypothesis Testing and Interpretation

The entire design is built to test these hypotheses:

Hypothesis for β₁ (Sentiment):

Null (H₀): β₁ = 0. Financial sentiment has no statistically significant impact on triggering future volatility.
Alternative (Hₐ): β₁ ≠ 0. Financial sentiment does have a significant impact.
Hypothesis for β₂ (Odds):

Null (H₀): β₂ = 0. Polymarket odds have no statistically significant impact.
Alternative (Hₐ): β₂ ≠ 0. Polymarket odds do have a significant impact.

## 4. The End Result: Quantified Insights

After running the estimation (typically via Maximum Likelihood Estimation in a stats package), you will get estimates for β₁ and β₂ along with their p-values.

If the p-value for β₁ is < 0.05, you can reject the null hypothesis and conclude sentiment is a significant driver.
The final step is to compare the absolute magnitudes of the significant coefficients. If your model yields β₁ = 0.6 and β₂ = 0.25, you have a quantitative result: "After controlling for baseline volatility and the influence of geopolitical odds, a one-standard-deviation increase in financial news sentiment has more than double the impact on exciting future volatility spikes."