## 1. Introduction: Beyond Linear Relationships
- **1.1. Core Concept:** Model the intensity (rate) of market volatility spikes, where each spike's ability to trigger future spikes depends on its characteristics (marks).

- **1.2. The Non-Parametric Advantage:** Move beyond simple linear assumptions ($\beta$ coefficients) to capture complex, non-linear interactions between event marks.

- **1.3. Role of the Neural Network:** Employ a Multi-Layer Perceptron (MLP) as a universal function approximator to learn the "influence function" `κ(m)` directly from the data, revealing its true shape.

## 2. The Modeling Workflow & Key Components
- 2.1. Data Synthesis (`synthesize_data`)
    - **Implied Volatility (IV):** A baseline series with randomly injected spikes to simulate market shocks.
    - **Marks (Features):**
        - **Financial Sentiment:** Designed to be generally anti-correlated with IV.
    - **Polymarket Odds Change:** Mostly noise, but with larger values coinciding with IV spikes.

## 2.2. Event Detection & Marking (get_events_and_marks)
**Event Trigger:** An "event" is defined when the Implied Volatility crosses a user-defined threshold (e.g., mean + 1.5 standard deviations).
**Mark Assignment:** Each event timestamp is "marked" with the corresponding sentiment and odds change values.
**Normalization:** Marks are standardized (zero mean, unit variance) to ensure the neural network treats both features fairly.

## 2.3. The Neural Network Estimator (train_nn_influence_model)
- **Objective:** Learn the mark influence function, `κ(m)`. This function maps a 2D mark vector m to a scalar value representing its influence.
- **Inputs:** The normalized 2D mark vectors (`sentiment_norm`, `odds_change_norm`).
- **Target (y):** A proxy for influence. For each event, the target is the number of subsequent events that occur within a "learning window" (defined as 1/α).
- **Architecture:** A simple MLP with a single hidden layer whose size is controlled by the user.

## 2.4. Final Intensity Calculation (plot_results)
The complete Hawkes process intensity λ(t) is calculated as:
$$\lambda(t)=\mu+\sum_{t_i<t}NN(m_i​)⋅\alpha \exp^{−\alpha(t−t_i​)}$$
Where `NN(m_i)` is the output of the trained neural network for the mark `m_i`.

## 3. Application UI & Interactive Elements
- 3.1. Sidebar Controls (The User's Input)
    - **Synthesize New Data Button:** Resets the simulation with a new random dataset.
    - **Volatility Threshold Slider:** Adjusts the sensitivity for event detection.
    - **$\mu$ (Baseline Intensity) Slider:** Sets the background rate of spontaneous events.
    - **$\alpha$ (Decay Speed) Slider:** Controls the memory of the process; how quickly an event's influence fades.
    - **Hidden Neurons Slider:** Adjusts the complexity of the neural network model.

## 3.2. Main Panel Displays (The Model's Output)
- Detected Event Data Table: A clear list of all data points that were classified as events based on the current threshold.
- **Plot 1:** Implied Volatility: Shows the raw IV series and highlights the detected event points.
- **Plot 2: Feature Marks:** Displays the raw sentiment and odds change time series.
- **Plot 3: Estimated Intensity $\lambda(t)$:** Visualizes the final Hawkes intensity, showing jumps after events.
- **Plot 4: Learned Influence Surface `κ(m)`:** The key non-parametric result. A 2D contour plot showing the learned influence for every combination of input marks.

## 4. How to Interpret the Results
- **4.1. Reading the Influence Surface (Contour Plot):**
    - **Color Scale:** Bright yellow areas correspond to mark combinations that are highly excitatory (trigger many future events). Dark purple areas have low or no influence.
    - **Axes:** The x-axis represents normalized sentiment, and the y-axis represents normalized odds change.

## 4.2. Identifying Non-Linear Patterns:
- **Gradients:** If color consistently brightens along one axis, it indicates a strong, almost-linear influence from that feature.
- **"Hotspots" & "Coldspots":** Look for isolated islands of bright yellow or dark purple. This reveals specific, non-linear conditions that are highly influential (e.g., "only events with very negative sentiment and very large odds changes are destabilizing").
- **Curvature:** A curved or saddle-shaped surface indicates complex interactions that a linear model would fail to capture.

## Flowchart
```mermaid
graph TD
    subgraph "1. User Configuration & Data Setup"
        A[Start] --> B(User sets parameters via sidebar <br> Threshold, μ, α, NN Neurons);
        B --> C{Synthesize New Data?};
        C -- Yes --> D[1.1. Synthesize Time Series Data <br> IV, Sentiment, Odds Change];
        C -- No --> E[Use Existing Data];
        D --> F[1.2. Detect Events & Assign Marks];
        E --> F;
        B --> F;
    end

    subgraph "2. Core Modeling & Analysis"
        F --> G{Are enough events detected?};
        G -- No --> H[Display Warning: "No events found, lower threshold"];
        G -- Yes --> I[Display Detected Event Data Table];
        I --> J[2.1. Train Neural Network <br> Input: Marks <br> Target: Subsequent event count];
    end

    subgraph "3. Visualization & Interpretation"
        J --> K[3.1. Calculate Final Intensity λ(t) <br> Uses μ, α, and the trained NN];
        J --> L[3.2. Generate Influence Surface Plot κ(m) <br> Shows the learned NN function];
        K & L --> M[Display All Plots <br> 1. Volatility & Events <br> 2. Feature Marks <br> 3. Hawkes Intensity <br> 4. Influence Surface];
        M --> N[4. User Interprets Results <br> Analyzes contour plot for non-linear patterns, <br> hotspots, and gradients];
    end

    subgraph "4. Interactive Loop"
        N --> B;
    end
```
