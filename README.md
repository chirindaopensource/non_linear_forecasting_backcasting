# `README.md`

# *Nonlinear Fore(Back)casting and Innovation Filtering for Causal-Noncausal VAR Models* Implementation
<br>

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-1674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checking: mypy](https://img.shields.io/badge/type_checking-mypy-blue.svg)](http://mypy-lang.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Joblib](https://img.shields.io/badge/Joblib-darkgreen.svg?style=flat&logo=python&logoColor=white)](https://joblib.readthedocs.io/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-blue.svg?style=flat&logo=python&logoColor=white)](https://www.statsmodels.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2205.09922-b31b1b.svg)](https://arxiv.org/abs/2205.09922)
[![Research](https://img.shields.io/badge/Research-Quantitative%20Finance-green)](https://github.com/chirindaopensource/non_linear_forecasting_backcasting)
[![Discipline](https://img.shields.io/badge/Discipline-Econometrics-blue)](https://github.com/chirindaopensource/non_linear_forecasting_backcasting)
[![Methodology](https://img.shields.io/badge/Methodology-Noncausal%20Time%20Series-orange)](https://github.com/chirindaopensource/non_linear_forecasting_backcasting)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/non_linear_forecasting_backcasting)
<br>

**Repository:** https://github.com/chirindaopensource/non_linear_forecasting_backcasting

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Nonlinear Fore(Back)casting and Innovation Filtering for Causal-Noncausal VAR Models"** by:

*   Christian Gourieroux
*   Joann Jasiak

The project provides a complete, computationally tractable system for the quantitative analysis of dynamic systems prone to speculative bubbles and other forms of locally explosive behavior. It enables robust, state-dependent risk assessment, probabilistic forecasting, and structural "what-if" scenario analysis that accounts for both nonlinear dynamics and model estimation uncertainty.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: run_full_research_pipeline](#key-callable-run_full_research_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the advanced econometric framework presented in Gourieroux and Jasiak (2025). The core of this repository is the iPython Notebook `non_linear_forecasting_backcasting_draft.ipynb`, which contains a comprehensive suite of functions to estimate, forecast, and analyze mixed causal-noncausal Vector Autoregressive (VAR) models.

Standard linear VAR models are purely causal and assume Gaussian errors, making them ill-suited for capturing the dynamics of financial and economic time series that exhibit bubbles, sudden crashes, or other forms of explosive behavior. The mixed causal-noncausal framework addresses this by allowing for roots of the VAR characteristic polynomial to lie both inside and outside the unit circle, generating a strictly stationary process with highly nonlinear, state-dependent dynamics.

This codebase enables researchers, quantitative analysts, and macroeconomists to:
-   Rigorously estimate mixed VAR models using the semi-parametric Generalized Covariance (GCov) method.
-   Generate full, non-Gaussian predictive densities for probabilistic forecasting.
-   Quantify estimation uncertainty using a novel backward-bootstrap procedure to create confidence sets for prediction intervals.
-   Filter the underlying nonlinear, structural innovations of the system.
-   Conduct state-dependent Impulse Response Function (IRF) analysis to understand how the system responds to shocks in "on-bubble" versus "off-bubble" states.

## Theoretical Background

The methodology implemented in this project is a direct translation of the unified framework presented in the source paper. It leverages the state-space representation of a VAR(p) process to separate its dynamics into stable (causal) and unstable (non-causal) components.

### 1. The Mixed Causal-Noncausal VAR Model

The model is defined by the standard VAR(p) equation, but with a critical difference in its assumptions:
$Y_t = \Phi_1 Y_{t-1} + \dots + \Phi_p Y_{t-p} + \epsilon_t$
The roots of the characteristic polynomial `det(I - \sum \Phi_i \lambda^i) = 0` can be both inside (`causal`) and outside (`non-causal`) the unit circle. The errors `\epsilon_t` are assumed to be i.i.d. and non-Gaussian.

### 2. State-Space Decomposition and Predictive Density

The VAR(p) process is transformed into a VAR(1) in state-space using the companion matrix `\Psi`. A **Jordan Decomposition** (`\Psi = A J A^{-1}`) separates the system into latent causal (`Z_1`) and non-causal (`Z_2`) states. This separation is the key to the paper's central theoretical result: a closed-form expression for the one-step-ahead predictive density, given in **Equation 3.1**:
$l(y | Y_T) = \frac{l_2(A^2 \tilde{y}_{T+1})}{l_2(A^2 \tilde{Y}_T)} |\det J_2| g(y - \sum \Phi_i Y_{T-i+1})$
- `g` is the density of the error `\epsilon_t`.
- `l_2` is the stationary density of the non-causal state `Z_2`.
This density is nonlinear and state-dependent, allowing it to capture complex dynamics.

### 3. Uncertainty Quantification via Backward Bootstrap

To account for estimation uncertainty, the framework uses a novel "backward bootstrap" procedure. Since the model is Markovian in both forward and reverse time, one can generate synthetic data paths by **backcasting** from the terminal observation `Y_T`. By re-estimating the model on many such paths, the sampling distribution of the prediction interval is obtained, which is then used to construct a robust **Confidence Set for the Prediction Interval (CSPI)**, as defined in **Equation 4.10**.

### 4. Nonlinear Innovation Filtering and State-Dependent IRFs

Standard VAR shocks are not meaningful in this context. The paper defines true, past-independent structural innovations `v_t` via the **Probability Integral Transform (PIT)**. This involves estimating the conditional CDF of the latent states and transforming it to a standard normal distribution.
**Equation 5.5:** $v_{2,t} = \Phi^{-1}[F_2(Z_{2,t}|Z_{t-1})]$
Simulating the model forward using the inverse of this transformation allows for the computation of **state-dependent Impulse Response Functions (IRFs)**, which show how the system's response to a shock `\delta` changes depending on its initial state (e.g., during a bubble).

## Features

The `non_linear_forecasting_backcasting_draft.ipynb` notebook implements the full research pipeline:

-   **Robust Data Pipeline:** Validation, cleaning, and preparation of time series data.
-   **Advanced Estimator:** A complete implementation of the semi-parametric GCov estimator for VAR parameters.
-   **Probabilistic Forecasting:** Functions to compute the full predictive density, point forecasts (mode), and prediction intervals.
-   **Advanced Uncertainty Quantification:** A parallelized implementation of the backward bootstrap with SIR sampling to generate confidence sets.
-   **Structural Analysis:** Functions to filter nonlinear innovations and simulate state-dependent IRFs.
-   **Model Validation:** A full simulation study framework to assess the finite-sample properties of the pipeline.
-   **Sensitivity Analysis:** Tools to conduct robustness checks on key model parameters.
-   **Integrated Visualization:** A dedicated class for generating all key publication-quality plots.

## Methodology Implemented

The codebase is a direct, one-to-one implementation of the paper's methodology:
1.  **Data Preparation (Tasks 1-2):** Ingests and prepares data as per the paper's empirical application.
2.  **Estimation (Tasks 3-5):** Implements the GCov estimator, Jordan decomposition, and non-parametric density estimation.
3.  **Forecasting (Tasks 6-8):** Implements the predictive density formula and extracts point and interval forecasts.
4.  **Uncertainty (Task 9):** Implements the full backward bootstrap with SIR sampling to compute confidence sets.
5.  **Structural Analysis (Tasks 10-11):** Implements the Nadaraya-Watson estimator for innovation filtering and the inverse for IRF simulation.
6.  **Validation & Orchestration (Tasks 12-17):** Provides high-level orchestrators for empirical analysis, simulation studies, robustness checks, and comparative analysis.

## Core Components (Notebook Structure)

The `non_linear_forecasting_backcasting_draft.ipynb` notebook is structured as a series of modular, professional-grade functions, each corresponding to a specific task in the pipeline. Key functions include:

-   **`validate_and_cleanse_data`**: The initial data quality gate.
-   **`prepare_var_data`**: Transforms data to be stationary and demeaned.
-   **`estimate_gcov_var`**: The core GCov estimation engine.
-   **`compute_jordan_decomposition`**: Separates causal/non-causal dynamics.
-   **`estimate_functional_components`**: Fits the non-parametric KDEs.
-   **`compute_predictive_density`**: The engine for probabilistic forecasting.
-   **`compute_point_forecast` & `compute_prediction_interval`**: Extracts forecast products.
-   **`compute_bootstrap_confidence_set`**: The advanced uncertainty quantification engine.
-   **`filter_nonlinear_innovations`**: Extracts structural shocks.
-   **`simulate_irf`**: Simulates state-dependent IRFs.
-   **`run_empirical_analysis`**: Orchestrates a full analysis of a single dataset.
-   **`run_full_research_pipeline`**: The single, top-level entry point to the entire project.

## Key Callable: run_full_research_pipeline

The central function in this project is `run_full_research_pipeline`. It orchestrates the entire analytical workflow from raw data to a final, comprehensive results dictionary.

```python
def run_full_research_pipeline(
    raw_df: pd.DataFrame,
    study_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes a complete, end-to-end research pipeline for the mixed
    causal-noncausal VAR model.
    ... (full docstring is in the notebook)
    """
    # ... (implementation is in the notebook)
```

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `statsmodels`, `joblib`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/non_linear_forecasting_backcasting.git
    cd non_linear_forecasting_backcasting
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies from `requirements.txt`:**
    ```sh
    pip install -r requirements.txt
    ```

## Input Data Structure

The primary input is a `pandas.DataFrame` with a monthly `DatetimeIndex` and two columns: `"real_oil_price"` and `"real_gdp"`.

**Example:**
```
                     real_gdp  real_oil_price
2019-04-30  18958.789123       63.870000
2019-05-31  19002.256789       60.210000
2019-06-30  19045.724455       57.430000
...                  ...             ...
```

## Usage

The entire pipeline is executed through the `run_full_research_pipeline` function. The user must provide the raw data and a comprehensive `study_params` dictionary that controls which analyses are run.

```python
import pandas as pd
import numpy as np

# 1. Load your data
# raw_data_df = pd.read_csv(...)
# For this example, we create synthetic data.
date_rng = pd.date_range(start='1986-01-01', end='2019-06-30', freq='M')
# ... (data generation code) ...
raw_data_df = pd.DataFrame(...)

# 2. Define your configurations (see notebook for full example)
study_params = {
    "run_empirical": {"enabled": True, ...},
    "run_simulation": {"enabled": False, ...},
    # ... other sections ...
}

# 3. Run the master pipeline
# from non_linear_forecasting_backcasting_draft import run_full_research_pipeline
# final_results = run_full_research_pipeline(
#     raw_df=raw_data_df,
#     study_params=study_params
# )

# 4. Instantiate the visualizer and plot results
# from non_linear_forecasting_backcasting_draft import ModelVisualizer
# visualizer = ModelVisualizer(final_results['empirical_analysis'])
# visualizer.plot_diagnostics()
# visualizer.plot_irf(irf_date=pd.Timestamp('2008-06-30'))
```

## Output Structure

The `run_full_research_pipeline` function returns a deeply nested dictionary containing all data artifacts. Top-level keys include:

-   `pipeline_configuration`: A copy of the input `study_params`.
-   `empirical_analysis`: Results from the core analysis on the provided data.
-   `simulation_study`: A DataFrame summarizing the Monte Carlo results.
-   `robustness_checks`: DataFrames detailing the sensitivity analysis.
-   `comparative_analysis`: A dictionary with forecast and metric DataFrames from the horse race.

## Project Structure

```
non_linear_forecasting_backcasting/
│
├── non_linear_forecasting_backcasting_draft.ipynb  # Main implementation notebook
├── requirements.txt                                # Python package dependencies
├── LICENSE                                         # MIT license file
└── README.md                                       # This documentation file
```

## Customization

The pipeline is highly customizable via the `study_params` dictionary. Users can easily modify:
-   The control flags (`run_empirical`, etc.) to enable or disable parts of the analysis.
-   The VAR lag order `p_lags`.
-   The GCov moment specifications `H_moment_lags` and `error_powers`.
-   All simulation parameters (`S_bootstrap_replications`, `n_baseline_sims`, etc.).
-   The specific dates for targeted forecasting and IRF analysis.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{gourieroux2022nonlinear,
  title={Nonlinear Fore(Back)casting and Innovation Filtering for Causal-Noncausal VAR Models},
  author={Gourieroux, Christian and Jasiak, Joann},
  journal={arXiv preprint arXiv:2205.09922},
  year={2022}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Python Implementation of the Gourieroux-Jasiak (2025) Framework for Causal-Noncausal VAR Models. 
GitHub repository: https://github.com/chirindaopensource/non_linear_forecasting_backcasting
```

## Acknowledgments

-   Credit to Christian Gourieroux and Joann Jasiak for their foundational theoretical and empirical work.
-   Thanks to the developers of the `pandas`, `numpy`, `scipy`, `matplotlib`, `statsmodels`, and `joblib` libraries, which provide the essential toolkit for this implementation.

--

This README was generated based on the structure and content of non_linear_forecasting_backcasting_draft.ipynb and follows best practices for research software documentation.
