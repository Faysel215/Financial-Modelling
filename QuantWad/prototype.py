import numpy as np
import time

class QuantumPathIntegrator:
    """
    A simplified 'Quantum' Path Integral (qPATHINT) engine for pricing
    path-dependent financial instruments.
    
    This simulates the evolution of the probability density function (PDF)
    of the asset price (or rate) over time using a grid-based propagator,
    analogous to solving the Schrödinger equation in imaginary time.
    """

    def __init__(self, s0, vol, r, t_max, grid_size=1000, sigma_range=5):
        """
        Args:
            s0: Initial price/rate.
            vol: Volatility (sigma).
            r: Risk-free rate (drift).
            t_max: Time to maturity (years).
            grid_size: Number of spatial points (N).
            sigma_range: How many standard deviations to cover in the grid.
        """
        self.s0 = s0
        self.vol = vol
        self.r = r
        self.t_max = t_max
        self.grid_size = grid_size
        
        # Log-space grid setup
        # Center grid around log(s0) + drift
        drift_total = (r - 0.5 * vol**2) * t_max
        center = np.log(s0) + drift_total
        width = sigma_range * vol * np.sqrt(t_max)
        
        self.x_min = center - width
        self.x_max = center + width
        self.dx = (self.x_max - self.x_min) / (grid_size - 1)
        self.x_grid = np.linspace(self.x_min, self.x_max, grid_size)
        
        # Initial Wave Function (Dirac Delta approximated as narrow Gaussian)
        # Represents the state being exactly at S0 at t=0
        self.psi = np.zeros(grid_size)
        initial_idx = np.abs(self.x_grid - np.log(s0)).argmin()
        self.psi[initial_idx] = 1.0 / self.dx  # Normalize to integral = 1
        
        print(f"Engine Initialized: Range[{np.exp(self.x_min):.4f}, {np.exp(self.x_max):.4f}]")

    def propagate(self, dt):
        """
        Evolve the wave function forward by time step dt using convolution.
        This corresponds to the free-particle propagator in the path integral.
        """
        # The Green's function (kernel) for Brownian motion in log-space
        # G(x, x') = (1 / sqrt(2*pi*var)) * exp(-(x - x' - mu*dt)^2 / (2*var))
        
        mu = self.r - 0.5 * self.vol**2
        variance = self.vol**2 * dt
        
        # Create the kernel grid (centered at 0)
        k_x = np.arange(-self.grid_size + 1, self.grid_size) * self.dx
        kernel = (1.0 / np.sqrt(2 * np.pi * variance)) * \
                 np.exp(-(k_x - mu * dt)**2 / (2 * variance))
        
        # Convolve current state with propagator
        # This is the numerical equivalent of the path integral: Psi(t+dt) = Integral(K * Psi(t))
        psi_new = np.convolve(self.psi, kernel, mode='same') * self.dx
        
        self.psi = psi_new

    def apply_wad_condition(self, k_strike, condition_type='call'):
        """
        Apply the Wa'd (Promise) logic. 
        In a path integral, this is a 'measurement' or 'potential' application.
        
        For a Wa'd Structure:
        If Spot > Strike, the promise is exercised (cash flow occurs).
        We calculate the expected value of this exercise at this specific time slice.
        """
        S = np.exp(self.x_grid)
        
        if condition_type == 'call':
            # Payoff = max(S - K, 0)
            payoff = np.maximum(S - k_strike, 0)
        elif condition_type == 'put':
            payoff = np.maximum(k_strike - S, 0)
            
        # Calculate expected value at this slice: Integral(Psi(x) * Payoff(x))
        expected_value = np.sum(self.psi * payoff) * self.dx
        return expected_value

    def price_wad_structure(self, time_steps, triggers):
        """
        Price a multi-leg Wa'd structure.
        
        Args:
            time_steps: Number of simulation steps.
            triggers: List of dicts [{'step': 100, 'k': 0.05}, ... ] representing
                      reset dates where a Wa'd might be exercised.
        """
        dt = self.t_max / time_steps
        total_value = 0.0
        
        print(f"Starting Propagation: {time_steps} steps...")
        start_time = time.time()
        
        trigger_map = {t['step']: t['k'] for t in triggers}
        
        for step in range(1, time_steps + 1):
            # 1. Propagate wave function
            self.propagate(dt)
            
            # 2. Check for Wa'd Trigger (Reset Date)
            if step in trigger_map:
                strike = trigger_map[step]
                
                # Calculate value of this specific leg
                leg_value = self.apply_wad_condition(strike)
                
                # Discount back to t=0
                discount_factor = np.exp(-self.r * (step * dt))
                present_leg_value = leg_value * discount_factor
                
                total_value += present_leg_value
                # Note: In a real IPRS, the contract continues. 
                # If it were a "Knock-out", we would zero out parts of self.psi here.
        
        elapsed = time.time() - start_time
        print(f"Pricing Complete. Time: {elapsed:.4f}s")
        return total_value

# --- Example Usage ---

if __name__ == "__main__":
    # Scenario: 5-Year Islamic Profit Rate Swap
    # We (Fixed Payer) promise to pay the difference if Floating > Fixed (0.05).
    # Modeled as a series of Caplets priced via Path Integral.
    
    # Initialize Engine
    # Spot Rate: 4.5%, Vol: 20%, RiskFree: 3%, 5 Years
    engine = QuantumPathIntegrator(s0=0.045, vol=0.2, r=0.03, t_max=5.0)
    
    # Define Wa'd trigger points (Quarterly resets for 5 years = 20 quarters)
    total_steps = 200 # 10 steps per quarter
    quarterly_resets = [i * 10 for i in range(1, 21)] 
    
    # Create trigger list: Fixed Rate is 5% (0.05)
    triggers = [{'step': s, 'k': 0.05} for s in quarterly_resets]
    
    # Run Pricing
    npv = engine.price_wad_structure(total_steps, triggers)
    
    print(f"\n--- Results ---")
    print(f"Structure: 5-Year IPRS (Fixed Payer Leg)")
    print(f"Method: Grid-Based Path Integral (qPATHINT approximation)")
    print(f"Calculated NPV of Floating Leg obligations: {npv:.6f}")

