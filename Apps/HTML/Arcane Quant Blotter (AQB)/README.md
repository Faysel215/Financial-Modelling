# Arcane Quant Blotter

Arcane Quant Blotter is a sophisticated, web-based user interface for quantitative options analysis. It provides a dense, professional, and highly functional dashboard for traders and analysts to visualize complex options data, including real-time option chains, implied volatility surfaces, and full Greek surfaces (Delta, Gamma, Vega, Theta).

The backend is powered by Python and simulates the use of neural networks for high-speed calculation of implied volatility and option greeks, demonstrating a modern quant workflow.

<!-- Placeholder - Replace with an actual screenshot -->

## Features

- **Live Option Chain:** Displays calls and puts side-by-side with key data points (IV, Delta, Bid/Ask, Volume, OI).
    
- **Interactive 3D Surface Plots:**
    
    - Implied Volatility (IV) Surface across strikes and expirations.
        
    - Full Greek Surfaces (Delta, Gamma, Vega, Theta).
        
- **2D IV Analysis Charts:**
    
    - **IV Skew:** Shows the "smile" or "smirk" for a given expiration.
        
    - **Term Structure:** Displays how volatility changes across different expiration dates.
        
- **High-Performance Backend:** Built with Python and FastAPI for asynchronous speed.
    
- **Simulated NN Engine:** Demonstrates the architecture for integrating pre-trained neural network models for complex calculations.
    
- **Clean, Modern UI:** Built with Tailwind CSS for a responsive and minimal design.
    

## Tech Stack

|   |   |
|---|---|
|**Component**|**Technology**|
|**Backend**|Python 3.9+, FastAPI, Uvicorn, NumPy, Pandas, Scikit-learn, Py_vollib|
|**Frontend**|HTML5, CSS3, Tailwind CSS, JavaScript, Chart.js (for 2D plots), Plotly.js (for 3D plots)|
|**Tooling**|Git, Python Virtual Environments|

## Project Structure

The project is organized into distinct `backend`, `frontend`, and `notebooks` directories to maintain a clean separation of concerns.

```
/arcane-quant-blotter/
│
├── 📁 backend/
│   ├── 📁 app/
│   │   ├── 📁 api/
│   │   │   └── endpoints.py
│   │   ├── 📁 core/
│   │   │   └── utils.py
│   │   ├── 📁 models/
│   │   │   └── nn_predictor.py
│   │   ├── 📁 services/
│   │   │   └── data_generator.py
│   │   └── main.py
│   ├── venv/
│   └── requirements.txt
│
├── 📁 frontend/
│   ├── 📄 index.html
│   └── 📁 js/
│       └── app.js
│
└── 📁 notebooks/
    └── 📄 (Jupyter notebooks for model training)
```

## Getting Started

Follow these instructions to get a local copy of the project up and running.

### Prerequisites

- Python 3.9 or newer
    
- `pip` and `venv` (usually included with Python)
    
- A modern web browser (Chrome, Firefox, Edge)
    

### Installation & Setup

1. **Clone the repository:**
    
    ```
    git clone https://github.com/your-username/arcane-quant-blotter.git
    cd arcane-quant-blotter
    ```
    
2. **Set up and activate the Python backend environment:**
    
    ```
    cd backend
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    
3. **Install backend dependencies:**
    
    ```
    pip install -r requirements.txt
    ```
    

### Running the Application

The application requires two components to be running simultaneously: the backend server and the frontend interface.

1. **Start the Backend Server:**
    
    - Make sure you are in the `backend` directory and your virtual environment is activated.
        
    - Run the Uvicorn server:
        
        ```
        uvicorn app.main:app --reload
        ```
        
    - The server should now be running at `http://127.0.0.1:8000`. Leave this terminal window open.
        
2. **Launch the Frontend:**
    
    - Open a new terminal window or navigate in your file explorer.
        
    - Go to the `frontend/` directory.
        
    - Simply open the `index.html` file in your web browser.
        

The application should load, and the frontend will automatically fetch data from the running backend to populate the blotter.

## Usage

- The blotter will load with synthetic data for a default ticker (e.g., TSLA).
    
- To load new synthetic data for a different ticker, type a symbol into the input box in the header and press **Enter**.
    

## Future Development

This project serves as a strong foundation. Future enhancements are outlined in Phase 5 of the Implementation Plan:

- **Train Real Models:** Use the `notebooks/` directory to train actual neural network models on historical options data.
    
- **Integrate Real Predictors:** Replace the simulated functions in `nn_predictor.py` with calls to the real, saved models.
    
- **Connect to Live Data:** Replace the synthetic data generator with a service that connects to a live financial data provider API (e.g., Polygon.io, Interactive Brokers).
    
- **Deployment:** Dockerize the backend and deploy the full application to a cloud environment.
    

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

## License

Distributed under the MIT License. See `LICENSE` for more information.