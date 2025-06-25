import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import streamlit as st

# --- 1. Quantum Walk Simulator Class ---
# This class handles the core logic of the Split-Step Quantum Walk (SSQW).

class QuantumWalkSimulator:
    """
    Simulates a 1D Split-Step Quantum Walk (SSQW).

    The state of the system is represented by a complex vector `psi` of shape
    (2, num_positions), where the first row corresponds to the spin-down
    amplitudes and the second row to the spin-up amplitudes at each position.
    """

    def __init__(self, num_steps: int):
        """
        Initializes the simulator.

        Args:
            num_steps: The total number of steps the walk will take.
        """
        self.num_steps = num_steps
        # The position space ranges from -num_steps to +num_steps.
        self.num_positions = 2 * num_steps + 1
        self.position_space = np.arange(-num_steps, num_steps + 1)
        # Initialize the state vector to all zeros.
        self.psi = np.zeros((2, self.num_positions), dtype=np.complex128)

    def set_initial_state(self, coin_state: np.ndarray, position_index: int = None):
        """
        Sets the initial state of the walker.

        Args:
            coin_state: A 2-element complex vector for the initial coin state.
            position_index: The starting index on the position lattice.
                            Defaults to the center of the lattice.
        """
        if position_index is None:
            position_index = self.num_steps  # Center of the array
        self.psi = np.zeros((2, self.num_positions), dtype=np.complex128)
        self.psi[0, position_index] = coin_state[0]
        self.psi[1, position_index] = coin_state[1]

    def _create_rotation_operator(self, theta: float) -> np.ndarray:
        """
        Creates the Ry(2*theta) rotation matrix for the coin.
        """
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return np.array([
            [cos_t, -sin_t],
            [sin_t,  cos_t]
        ], dtype=np.complex128)

    def _apply_step(self, R1: np.ndarray, R2: np.ndarray):
        """
        Applies one full step of the SSQW evolution.
        U = S_down * R2 * S_up * R1
        """
        # 1. Apply first rotation R1
        self.psi = np.einsum('ij,jk->ik', R1, self.psi)

        # 2. Apply spin-up shift (S_up)
        self.psi[1, :] = np.roll(self.psi[1, :], 1)

        # 3. Apply second rotation R2
        self.psi = np.einsum('ij,jk->ik', R2, self.psi)

        # 4. Apply spin-down shift (S_down)
        self.psi[0, :] = np.roll(self.psi[0, :], -1)

    def run_walk(self, params: list):
        """
        Executes the full quantum walk for a given set of parameters.

        Args:
            params: A list containing the coin parameters [theta1, theta2].
        """
        theta1, theta2 = params
        R1 = self._create_rotation_operator(theta1)
        R2 = self._create_rotation_operator(theta2)

        for _ in range(self.num_steps):
            self._apply_step(R1, R2)

    def get_probability_distribution(self) -> np.ndarray:
        """
        Calculates the final probability distribution over all positions.
        """
        return np.abs(self.psi[0, :])**2 + np.abs(self.psi[1, :])**2

# --- 2. Variational Fitter Class ---
# This class manages the hybrid quantum-classical optimization loop.

class VariationalFitter:
    """
    Uses a classical optimizer to find the best quantum walk parameters
    to fit a target probability distribution.
    """

    def __init__(self, target_distribution: np.ndarray, num_steps: int):
        if len(target_distribution) != 2 * num_steps + 1:
            raise ValueError("Target distribution size must match the walk's position space.")
        
        self.target_distribution = target_distribution
        self.num_steps = num_steps
        self.q_walk = QuantumWalkSimulator(num_steps)
        self.initial_coin_state = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])

    def _cost_function(self, params: list) -> float:
        self.q_walk.set_initial_state(self.initial_coin_state)
        self.q_walk.run_walk(params)
        generated_distribution = self.q_walk.get_probability_distribution()
        
        # Ensure normalization for fair comparison
        sum_gen = np.sum(generated_distribution)
        if sum_gen > 1e-9:
             generated_distribution /= sum_gen
        
        cost = np.mean((self.target_distribution - generated_distribution)**2)
        return cost

    def fit(self, initial_params: list, method='Nelder-Mead', progress_bar=None) -> dict:
        """
        Runs the classical optimization routine.
        """
        global iteration_count
        iteration_count = 0
        
        def callback(xk):
            global iteration_count
            iteration_count += 1
            if progress_bar and max_iter > 0:
                progress_bar.progress(iteration_count / max_iter, text=f"Optimizing... Iteration {iteration_count}")

        max_iter = 200 # Default max iterations for progress bar
        
        result = minimize(
            self._cost_function,
            initial_params,
            method=method,
            options={'maxiter': max_iter, 'adaptive': True},
            callback=callback if progress_bar else None
        )
        if progress_bar:
            progress_bar.progress(1.0, text="Optimization Complete!")
        return result

# --- 3. Data Generation and Plotting ---

def generate_dummy_bimodal_data(x: np.ndarray, mu1: float, sigma1: float, mu2: float, sigma2: float, weight1: float = 0.5) -> np.ndarray:
    """
    Generates a normalized bimodal (two-peaked) distribution.
    """
    def gaussian(x_vals, mu, sigma):
        if sigma <= 0: return np.zeros_like(x_vals)
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mu) / sigma)**2)

    dist1 = gaussian(x, mu1, sigma1)
    dist2 = gaussian(x, mu2, sigma2)
    
    combined_dist = weight1 * dist1 + (1 - weight1) * dist2
    
    # Normalize to ensure it's a valid probability distribution
    sum_dist = np.sum(combined_dist)
    if sum_dist > 1e-9:
        return combined_dist / sum_dist
    return np.zeros_like(x)

# --- 4. Streamlit User Interface ---

def main():
    st.set_page_config(layout="wide")
    st.title("Interactive Multi-Split-Step Quantum Walk Simulator")
    st.write("Use the sidebar to control simulation parameters and generate a target financial distribution. The algorithm will then find the optimal quantum walk parameters to match it.")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("Simulation Controls")

        NUM_STEPS = st.slider("Number of Quantum Walk Steps", min_value=10, max_value=100, value=50, step=5)
        
        st.subheader("Target Distribution Parameters")
        st.write("Define the shape of the 'financial data' to model.")
        
        col1, col2 = st.columns(2)
        with col1:
            mu1 = st.slider("Peak 1 Position (μ₁)", -float(NUM_STEPS), float(NUM_STEPS), -20.0, 0.5)
            sigma1 = st.slider("Peak 1 Spread (σ₁)", 1.0, float(NUM_STEPS)/2, 8.0, 0.5)
        with col2:
            mu2 = st.slider("Peak 2 Position (μ₂)", -float(NUM_STEPS), float(NUM_STEPS), 15.0, 0.5)
            sigma2 = st.slider("Peak 2 Spread (σ₂)", 1.0, float(NUM_STEPS)/2, 12.0, 0.5)
        
        weight1 = st.slider("Peak 1 Weight", 0.0, 1.0, 0.6, 0.05)

        st.subheader("Optimizer Settings")
        run_button = st.button("Run Optimization", type="primary")

    # --- Main Panel ---
    POSITION_SPACE = np.arange(-NUM_STEPS, NUM_STEPS + 1)

    # Generate and display the target distribution
    target_dist = generate_dummy_bimodal_data(
        x=POSITION_SPACE,
        mu1=mu1, sigma1=sigma1,
        mu2=mu2, sigma2=sigma2,
        weight1=weight1
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(POSITION_SPACE, target_dist, 'r-', lw=2.5, label='Target Distribution (Dummy Data)')
    ax.set_title('Target Distribution Shape', fontsize=16)
    ax.set_xlabel('Position (Asset Return Bucket)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plot_placeholder = st.empty()
    plot_placeholder.pyplot(fig)

    if run_button:
        # --- Run the Variational Fitting ---
        fitter = VariationalFitter(target_distribution=target_dist, num_steps=NUM_STEPS)
        
        # Randomize the initial guess to avoid getting stuck in the same local minimum
        initial_guess = np.random.rand(2) * 2 * np.pi
        
        # UI for progress
        st.subheader("Optimization Results")
        st.info(f"Starting optimization with random initial guess: θ₁={initial_guess[0]:.4f}, θ₂={initial_guess[1]:.4f}")
        progress_bar = st.progress(0, text="Starting optimization. Please wait...")
        
        # Find the optimal parameters
        optimization_result = fitter.fit(initial_params=initial_guess, progress_bar=progress_bar)
        best_params = optimization_result.x
        final_cost = optimization_result.fun
        
        # --- Visualize the Final Results ---
        final_walk = QuantumWalkSimulator(num_steps=NUM_STEPS)
        final_walk.set_initial_state(fitter.initial_coin_state)
        final_walk.run_walk(best_params)
        final_distribution = final_walk.get_probability_distribution()
        
        sum_final = np.sum(final_distribution)
        if sum_final > 1e-9:
            final_distribution /= sum_final

        # Update plot with the generated distribution
        ax.bar(POSITION_SPACE, final_distribution, width=0.8, alpha=0.7, label='Generated SSQW Distribution')
        ax.legend()
        ax.set_title('Multi-SSQW Fitting Result', fontsize=16)
        plot_placeholder.pyplot(fig)

        # Display final metrics
        col1_res, col2_res, col3_res = st.columns(3)
        col1_res.metric("Optimal θ₁", f"{best_params[0]:.4f}")
        col2_res.metric("Optimal θ₂", f"{best_params[1]:.4f}")
        col3_res.metric("Final Mean Squared Error", f"{final_cost:.6f}")

if __name__ == '__main__':
    # To run this app, save it as a Python file (e.g., app.py)
    # and run the following command in your terminal:
    # streamlit run app.py
    main()
