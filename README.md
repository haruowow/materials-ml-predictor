# Materials Property Predictor 🔬

Machine learning system for predicting materials properties (band gap, formation energy) from chemical composition. Built for materials discovery acceleration using Materials Project data.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project demonstrates:
- **Data Collection**: Automated Materials Project API integration
- **Feature Engineering**: 98 composition-based features from chemical formulas
- **Machine Learning**: 4 models (Ridge, Random Forest, XGBoost, Neural Network)
- **Web Interface**: Interactive Streamlit app for predictions
- **Model Analysis**: Performance evaluation and limitations discovery

## 📊 Results

### Model Performance (Band Gap Prediction)

| Model | Test R² | Test MAE (eV) | Best For |
|-------|---------|---------------|----------|
| Ridge Regression | ~0.83 | ~0.48 | Baseline |
| Random Forest | ~0.89 | ~0.35 | Metals & Semiconductors |
| XGBoost | ~0.90 | ~0.32 | Overall Performance |
| Neural Network | ~0.91 | ~0.30 | Complex Patterns |

### Example Predictions

| Material | Random Forest | Actual | Error |
|----------|---------------|--------|-------|
| Si (semiconductor) | 0.900 eV | 1.1 eV | 0.2 eV |
| Fe (metal) | 0.504 eV | 0.0 eV | 0.5 eV |
| TiO2 (semiconductor) | 2.613 eV | 3.0 eV | 0.4 eV |

## 🔧 Tech Stack

- **ML/Data Science**: PyTorch, scikit-learn, XGBoost, pandas, NumPy
- **Materials Science**: pymatgen, matminer, Materials Project API
- **Visualization**: matplotlib, seaborn, Plotly
- **Web App**: Streamlit
- **Development**: Jupyter notebooks

## 🚀 Quick Start

### Prerequisites
\\\ash
Python 3.8+
Materials Project API key (free from materialsproject.org)
\\\

### Installation
\\\ash
# Clone repository
git clone https://github.com/yourusername/materials_ml_predictor.git
cd materials_ml_predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export MP_API_KEY='your_api_key_here'
\\\

### Usage
\\\ash
# 1. Collect data
python src/data_collection.py

# 2. Feature engineering
python src/preprocessing.py

# 3. Train models
python src/training.py

# 4. Launch web app
streamlit run web_app/app.py
\\\

## 📁 Project Structure

\\\
materials_ml_predictor/
├── src/
│   ├── data_collection.py    # Materials Project API interface
│   ├── preprocessing.py       # Feature engineering
│   └── training.py           # Model training & evaluation
├── web_app/
│   └── app.py                # Streamlit interface
├── notebooks/
│   └── 01_data_exploration.ipynb
├── requirements.txt
└── README.md
\\\

## 🔍 Key Findings

### Model Performance
- **Random Forest** excels at metals (0 eV) and semiconductors (1-3 eV)
- **Neural Network** achieves highest overall R² (~0.91)
- **Ridge Regression** struggles with non-linear relationships

### Dataset Analysis
- Training data: 1000 materials from Materials Project
- **Imbalance identified**: 78.7% metals, only 4.1% insulators
- Models underpredict wide band gap materials (>5 eV)

### Proposed Improvements
1. Stratified sampling for balanced dataset
2. Class weighting for rare material types
3. Separate models for different band gap ranges
4. Transfer learning from larger databases

## 💡 Future Work

- [ ] Graph Neural Networks for crystal structure
- [ ] Multi-task learning (predict multiple properties)
- [ ] Active learning integration with DFT
- [ ] Uncertainty quantification
- [ ] API deployment (FastAPI/Flask)

## 📊 Example Usage

\\\python
from preprocessing import MaterialsFeatureEngineer
import joblib

# Load trained model
model = joblib.load('models/random_forest.pkl')
engineer = MaterialsFeatureEngineer()
engineer.load_preprocessor('models')

# Predict band gap
formula = "Si"
features = engineer.create_features(formula)
band_gap = model.predict(features)
print(f"Predicted band gap for {formula}: {band_gap:.2f} eV")
\\\

## 📚 References

- [Materials Project](https://materialsproject.org/) - Data source
- [pymatgen](https://pymatgen.org/) - Materials analysis library
- [matminer](https://hackingmaterials.lbl.gov/matminer/) - Feature engineering

## 📝 License

MIT License - see LICENSE file

## 🤝 Acknowledgments

- Materials Project for providing open materials data
- Anthropic for Claude AI assistance in development

## 👤 Author

Your Name - [GitHub](https://github.com/yourusername)

---

**Note**: This project was developed as part of my MIT application portfolio, demonstrating machine learning, materials science, and software engineering skills.
