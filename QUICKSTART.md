# Quick Start Guide - Materials Property Predictor

This guide will help you get started with the Materials Property Predictor project.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Materials Project API key (free from https://materialsproject.org/api)

## Installation

### 1. Set up Python environment (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install dependencies

```bash
cd materials_ml_predictor
pip install -r requirements.txt
```

This will install all required packages including:
- PyTorch for neural networks
- scikit-learn for traditional ML
- pymatgen for materials science
- matminer for feature engineering
- Streamlit for web interface

## Getting Your API Key

1. Go to https://materialsproject.org
2. Create a free account
3. Navigate to your Dashboard
4. Copy your API key

## Running the Project

### Step 1: Data Collection

Set your API key and collect data:

```bash
export MP_API_KEY='your_api_key_here'  # Linux/Mac
# or
set MP_API_KEY=your_api_key_here  # Windows

python src/data_collection.py
```

This will download ~5000 materials with band gap data from Materials Project.

**Expected output:**
- `data/band_gaps.csv` - Raw materials data
- Console output showing data statistics

**Time:** ~5-10 minutes depending on connection

### Step 2: Feature Engineering

Generate ML features from chemical formulas:

```bash
python src/preprocessing.py
```

This converts chemical formulas into numerical features that ML models can use.

**Expected output:**
- `data/processed/band_gaps_with_features.csv` - Data with ML features
- `data/processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` - Train/test splits
- `models/scaler.pkl`, `models/feature_names.pkl` - Preprocessing artifacts

**Time:** ~5-15 minutes for 5000 materials

### Step 3: Model Training

Train multiple ML models:

```bash
python src/training.py
```

This trains and evaluates:
- Ridge & Lasso Regression (baselines)
- Random Forest
- XGBoost
- Neural Network

**Expected output:**
- `models/*.pkl` - Trained models
- `results/model_comparison.json` - Performance metrics
- `results/figures/*.png` - Visualizations
- Console output showing training progress and final comparison

**Time:** ~10-30 minutes depending on your hardware

### Step 4: Launch Web Interface

Start the interactive web app:

```bash
streamlit run web_app/app.py
```

Your browser should automatically open to http://localhost:8501

**Features:**
- Single material prediction
- Batch prediction from CSV
- Model comparison
- Interactive visualizations

## Expected Results

After training, you should see results similar to:

```
Model Comparison Summary
========================
                  train_mae  train_rmse  train_r2  test_mae  test_rmse  test_r2
Ridge Regression     0.45       0.62      0.85      0.48      0.65      0.83
Random Forest        0.15       0.25      0.97      0.35      0.52      0.89
XGBoost              0.20       0.32      0.95      0.32      0.48      0.90
Neural Network       0.18       0.28      0.96      0.30      0.45      0.91

🏆 Best Model (by Test R²): Neural Network (R² = 0.91)
```

Note: Actual results will vary based on the data downloaded and training parameters.

## Exploring with Jupyter

Launch Jupyter to explore the data:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook includes:
- Data visualization
- Statistical analysis
- Feature correlations
- Material examples

## Troubleshooting

### API Key Issues

If you get "No API key found":
```bash
# Check if environment variable is set
echo $MP_API_KEY  # Linux/Mac
echo %MP_API_KEY%  # Windows
```

### Installation Issues

**PyTorch not installing:**
Visit https://pytorch.org and get the correct install command for your system.

**Pymatgen errors:**
Try installing with conda:
```bash
conda install -c conda-forge pymatgen
```

### Memory Issues

If training crashes due to memory:
- Reduce dataset size in `data_collection.py` (change `max_materials`)
- Reduce batch size in `training.py` for neural network
- Use fewer trees in Random Forest

### Slow Performance

**Data collection slow:**
- Reduce `max_materials` parameter
- Check your internet connection

**Training slow:**
- Reduce number of estimators in Random Forest/XGBoost
- Reduce epochs for neural network
- Use GPU if available (PyTorch will auto-detect)

## Project Customization

### Predict Different Properties

To predict formation energy instead of band gap:

1. In `data_collection.py`, use `collect_formation_energy_data()`
2. Update `target_column` in `preprocessing.py` to `'formation_energy_per_atom'`
3. Retrain models

### Add More Features

Edit `preprocessing.py` to add custom features:
- Use matminer's advanced featurizers
- Add domain-specific features
- Include crystal structure information

### Tune Models

In `training.py`, modify hyperparameters:
```python
# Example: Better Random Forest
trainer.train_random_forest(
    n_estimators=200,  # More trees
    max_depth=30,      # Deeper trees
)

# Example: Bigger Neural Network
trainer.train_neural_network(
    hidden_dims=[512, 256, 128, 64],  # More layers/neurons
    epochs=200,                        # Longer training
)
```

## Next Steps for Your Portfolio

### Documentation
1. Write a detailed README with your results
2. Document your methodology
3. Explain model choices and performance

### Extensions
1. **Advanced features:** Implement graph neural networks for crystal structures
2. **Multi-task learning:** Predict multiple properties simultaneously
3. **Uncertainty quantification:** Add prediction confidence intervals
4. **Transfer learning:** Fine-tune on smaller experimental datasets
5. **Interpretability:** Use SHAP values to explain predictions

### Presentation
1. Create visualizations of your best results
2. Show error analysis (where models fail and why)
3. Compare to literature baselines
4. Discuss real-world applications

### GitHub Repository
Structure your GitHub repo professionally:
```
README.md - Project overview, results, instructions
docs/ - Extended documentation
examples/ - Example predictions and use cases
tests/ - Unit tests for your code
.gitignore - Exclude data and model files
requirements.txt - Pin exact versions
LICENSE - Choose appropriate license
```

## Resources

- **Materials Project API:** https://docs.materialsproject.org
- **Matminer documentation:** https://hackingmaterials.lbl.gov/matminer/
- **Pymatgen tutorials:** https://pymatgen.org
- **Materials ML papers:** Search "machine learning materials property prediction" on arXiv

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify all installation steps were completed
3. Ensure your API key is set correctly
4. Try with a smaller dataset first

Good luck with your project! This is a solid foundation for an MIT-level portfolio piece.
