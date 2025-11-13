# Quick Start Guide

Get up and running with Materials ML Predictor in 10 minutes!

## ⚡ Prerequisites

- Python 3.8 or higher
- Materials Project API key ([Get one free here](https://materialsproject.org/api))

## 🚀 Installation (5 minutes)

### 1. Clone and Setup
```bash
# Clone repository
git clone https://github.com/haruowow/materials-ml-predictor.git
cd materials-ml-predictor

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
venv\Scripts\activate
# Windows CMD:
venv\Scripts\activate.bat
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set API Key
```bash
# Windows PowerShell:
$env:MP_API_KEY = "your_api_key_here"

# Windows CMD:
set MP_API_KEY=your_api_key_here

# Mac/Linux:
export MP_API_KEY='your_api_key_here'
```

## 🎯 Usage (5 minutes)

### Option 1: Use Pre-trained Models (If Available)

If someone shared trained models with you:
```bash
# Just launch the web app
streamlit run web_app/app.py
```

Open http://localhost:8501 and start predicting!

### Option 2: Train Your Own Models
```bash
# Step 1: Collect data (3-5 minutes)
python src/data_collection.py
# Choose option 1 (Regular dataset)

# Step 2: Process features (1-2 minutes)
python src/preprocessing.py

# Step 3: Train models (10-15 minutes)
python src/training.py

# Step 4: Launch web app
streamlit run web_app/app.py
```

## 🧪 Try It Out!

Once the web app is running:

1. **Enter a chemical formula**: Try `Si`, `GaAs`, or `TiO2`
2. **See predictions**: Get band gap predictions from multiple models
3. **Compare models**: Random Forest is usually most accurate
4. **Batch predictions**: Upload a CSV file with multiple formulas

### Expected Results

| Material | Your Prediction | Literature Value |
|----------|----------------|------------------|
| Si | ~0.9 eV | 1.14 eV |
| GaAs | ~1.1 eV | 1.42 eV |
| TiO2 | ~2.6 eV | 3.00 eV |
| GaN | ~2.9 eV | 3.39 eV |

Typical accuracy: **±0.35 eV**

## 📊 Quick Python API

Use the models in your own code:
```python
import pandas as pd
import joblib
from src.preprocessing import MaterialsFeatureEngineer

# Load models
rf_model = joblib.load('models/random_forest.pkl')
xgb_model = joblib.load('models/xgboost.pkl')
engineer = MaterialsFeatureEngineer()
engineer.load_preprocessor('models')

# Predict for a material
formula = 'GaAs'
df = pd.DataFrame({'formula': [formula]})
df_features = engineer.create_simple_features(df)
X = df_features[engineer.feature_names].fillna(0).values
X_scaled = engineer.scaler.transform(X)

# Get predictions
rf_pred = rf_model.predict(X_scaled)[0]
xgb_pred = xgb_model.predict(X_scaled)[0]
ensemble = (rf_pred + xgb_pred) / 2

print(f"{formula} band gap predictions:")
print(f"  Random Forest: {rf_pred:.2f} eV")
print(f"  XGBoost: {xgb_pred:.2f} eV")
print(f"  Ensemble: {ensemble:.2f} eV")
```

## 🎯 Model Performance

Our models achieve:
- **R² Score**: 92.4% (Random Forest)
- **MAE**: 0.35 eV (Ensemble average)
- **Prediction Time**: <100ms per material

Best model: **Random Forest** (most reliable)

## 🔧 Troubleshooting

### "No module named 'mp_api'"
```bash
pip install mp-api
```

### "API key not found"
Make sure you set the environment variable:
```bash
# Check if it's set
echo $env:MP_API_KEY  # Windows PowerShell
echo $MP_API_KEY      # Mac/Linux
```

### "Models not found" in web app
You need to train the models first:
```bash
python src/data_collection.py
python src/preprocessing.py
python src/training.py
```

### Predictions seem wrong
- Check that you're using the **Random Forest** prediction (most accurate)
- Ensemble should be within 0.3-0.5 eV of literature values
- Some materials may be outside the training distribution

## 📚 Next Steps

- **Read the full documentation**: Check out START_HERE.md for detailed explanations
- **Explore notebooks**: `notebooks/01_data_exploration.ipynb` has data analysis
- **Customize models**: Edit `src/training.py` to try different hyperparameters
- **Add more data**: Increase dataset size in `src/data_collection.py`

## 🎓 Quick Tips

✅ **Always trust Random Forest first** - it's the most reliable
✅ **Ensemble is good for safety** - averages out errors
✅ **Expect ±0.35 eV error** - this is normal for ML models
✅ **Test with common materials** - Si, GaAs, TiO2, GaN work best

## 💡 Common Use Cases

1. **Materials Screening**: Quickly filter thousands of candidates
2. **Educational Demo**: Show students how ML works in science
3. **Research Tool**: Get initial estimates before DFT calculations
4. **Data Exploration**: Understand structure-property relationships

## 🚀 You're Ready!

You now have a working materials property predictor. Start exploring and making predictions!

Need help? Check START_HERE.md or open an issue on GitHub.

---

**Happy Predicting! 🔬**