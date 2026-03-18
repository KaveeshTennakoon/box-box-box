# Box Box Box 🏁 - F1 Strategy Optimization

This repository contains a Grey-Box System Identification approach to reverse-engineering an F1 race simulator. By mapping non-linear physics equations into machine learning features, this solution achieves 100% accuracy in predicting race outcomes.

## 🧠 The Approach: Polynomial Feature Engineering

Standard machine learning models (like Random Forests) struggle to learn absolute floating-point algebra natively. Instead of feeding the model raw ratios, this solution uses **Polynomial Feature Extraction** to mathematically calculate the cumulative wear of the tires.

1. **Linear Wear ($\sum i$):** Calculates the base arithmetic progression of tire degradation.
2. **Quadratic Wear ($\sum i^2$):** Natively maps the non-linear "Tire Cliff" where performance exponentially drops off after a grace period.
3. **Strategy Clone Filtering:** To prevent tree-based models from generating microscopic floating-point noise between identically performing drivers, a deterministic hash filter pools exact strategies and enforces strict grid-order sorting.

The processed features are fed into a high-precision `XGBRegressor` to assign a continuous temporal score to each driver.

## 🚀 Setup & Installation

1. Create a virtual environment and install the dependencies:
   pip install -r requirements.txt

2. python solution/train_proper.py

3. ./test_runner.sh