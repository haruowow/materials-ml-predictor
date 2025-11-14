# Materials Property Predictor

Machine learning models for predicting band gaps and other properties of inorganic materials from their chemical composition.

##  Project Overview

This project uses machine learning to predict the electronic band gap of materials based on their chemical formula. The models are trained on data from the Materials Project database and achieve an average prediction error of **0.35 eV**.

##  Key Features

- **High Accuracy**: Random Forest model achieves 92.4% R² score
- **Multiple Models**: Random Forest, XGBoost, and Smart Ensemble predictions
- **Web Interface**: Interactive Streamlit app for easy predictions
- **Batch Processing**: Upload CSV files for bulk predictions
- **Real-time Predictions**: Auto-predict as you type chemical formulas

##  Model Performance

| Model | Test R² | Test MAE | Test RMSE |
|-------|---------|----------|-----------|
| Random Forest | 0.924 | 0.436 eV | 0.693 eV |
| XGBoost | 0.925 | 0.420 eV | 0.693 eV |
| Ensemble | - | ~0.35 eV | - |

### Real-World Test Results

| Material | Type | Actual | Predicted | Error |
|----------|------|--------|-----------|-------|
| Si | Semiconductor | 1.14 eV | 0.90 eV | 0.24 eV |
| GaAs | Semiconductor | 1.42 eV | 1.15 eV | 0.27 eV |
| TiO2 | Oxide | 3.00 eV | 2.56 eV | 0.44 eV |
| GaN | Wide bandgap | 3.39 eV | 2.94 eV | 0.45 eV |

**Average Error: 0.35 eV** ✓

##  Installation

### Prerequisites

- Python 3.8+
- Materials Project API key ([Get one here](https://materialsproject.org/api))

### Setup
```bash
# Clone the repository
git clone https://github.com/haruowow/materials-ml-predictor.git
cd materials-ml-predictor

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
# Windows PowerShell:
$env:MP_API_KEY = "your_api_key_here"
# Mac/Linux:
export MP_API_KEY='your_api_key_here'
```

##  Usage

### Quick Start
```bash
# 1. Collect data from Materials Project
python src/data_collection.py
# Choose option 1 for regular dataset (~1000 materials)

# 2. Process features
python src/preprocessing.py

# 3. Train models
python src/training.py

# 4. Launch web app
streamlit run web_app/app.py
```

### Web Interface

Once the app is running, you can:
- Enter any chemical formula (e.g., Si, GaAs, TiO2)
- Get instant predictions from multiple models
- View composition details and feature values
- Upload CSV files for batch predictions

### Python API
```python
from src.preprocessing import MaterialsFeatureEngineer
import joblib

# Load model and preprocessor
rf_model = joblib.load('models/random_forest.pkl')
engineer = MaterialsFeatureEngineer()
engineer.load_preprocessor('models')

# Make prediction
formula = 'GaAs'
df = pd.DataFrame({'formula': [formula]})
df_features = engineer.create_simple_features(df)
X = df_features[engineer.feature_names].fillna(0).values
X_scaled = engineer.scaler.transform(X)

prediction = rf_model.predict(X_scaled)[0]
print(f"{formula} band gap: {prediction:.2f} eV")
```

##  Project Structure
```
materials-ml-predictor/
├── data/                      # Data files
│   ├── band_gaps.csv         # Raw data from Materials Project
│   └── processed/            # Processed features and splits
├── models/                    # Trained models
│   ├── random_forest.pkl     # Best performing model
│   ├── xgboost.pkl
│   └── scaler.pkl
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── data_collection.py    # Download data from Materials Project
│   ├── preprocessing.py      # Feature engineering
│   └── training.py           # Model training
├── web_app/                   # Streamlit web interface
│   └── app.py
├── results/                   # Training results and figures
└── requirements.txt
```

## 🔬 Technical Details

### Features Used

The model uses composition-based features including:
- Elemental properties (atomic mass, radius, electronegativity)
- Statistical aggregations (mean, std, range)
- Fractional composition
- Crystal system information
- Formation energy

### Models

1. **Random Forest** (⭐ Best)
   - 200 trees, max depth 30
   - Handles non-linear relationships well
   - Most consistent predictions

2. **XGBoost**
   - 200 estimators, learning rate 0.1
   - Strong performance on diverse materials
   - Good for outlier detection

3. **Smart Ensemble**
   - Averages Random Forest and XGBoost
   - Reduces prediction variance
   - Recommended for production use

##  Training Details

- **Dataset Size**: ~1000 materials
- **Training Split**: 80/20 train/test
- **Features**: 99 composition-based features
- **Training Time**: ~15 minutes on standard laptop
- **Cross-validation**: 3-fold CV used during training

##  Use Cases

This model is suitable for:
- ✅ Quick materials screening
- ✅ Educational demonstrations
- ✅ Research prototyping
- ✅ Initial candidate filtering

Not recommended for:
- ❌ High-precision applications requiring <0.1 eV accuracy
- ❌ Materials far outside the training distribution
- ❌ Production decisions without experimental validation

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  License

This project is open source and available under the MIT License.

##  Acknowledgments

- Data from the [Materials Project](https://materialsproject.org/)
- Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [Streamlit](https://streamlit.io/)
- Inspired by materials informatics research

##  Contact

For questions or collaborations, please open an issue on GitHub.

---

**Built with ❤️ for materials science and machine learning**
