# 🚀 AutoML Studio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://localhost:8501)

AutoML Studio is a modular, professional-grade automated machine learning platform designed for rapid model development, evaluation, and explainability.

---

## ✨ Key Features

- **📂 Seamless Data Handling**: Multi-modal data support including Numerical, Categorical, and Date-Time features.
- **⚡ Automated Feature Engineering**: 
    - Automatic detection of Task Type (Classification/Regression).
    - Intelligent handling of **High-Cardinality** features using Binary Encoding.
    - Automated **Date-Time extraction** (Year, Month, Day, Hour, Day of Week).
- **🚀 Model Registry & Tuning**: Automatically compares multiple models (Random Forest, Gradient Boosting, etc.) and performs hyperparameter tuning via GridSearch.
- **🔍 Model Explainability**: Integrated **SHAP (SHapley Additive exPlanations)** to provide transparency and trust in model predictions.
- **🧪 Robust Testing Suit**: Unit tests for core preprocessing and feature engineering logic ensuring reliability.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/automl-studio.git
cd automl-studio

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -e .
# Or
pip install -r requirements.txt
```

## 🚀 Usage

### UI Mode
Start the interactive dashboard:
```bash
streamlit run ui/app.py
```

### CLI Mode
Run the pipeline from the terminal:
```bash
python run_automl.py --dataset data/raw/sample.csv
```

---

## 🧪 Testing

We value code quality. Run the test suite using `pytest`:
```bash
pytest tests/
```

---

## 🗺️ GSoC 2026 Roadmap

This project is built with Google Summer of Code standards in mind. Future enhancements include:
- [ ] Integration with MLflow for experiment tracking.
- [ ] Support for Time-Series forecasting.
- [ ] Deployment as a REST API using the existing FastAPI scaffold.
- [ ] Advanced Model Explainability (permutation importance, Partial Dependence Plots).

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
