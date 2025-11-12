# Materials Property Predictor

Machine learning models for predicting band gaps and other properties of materials from their chemical formulas. Uses data from the Materials Project database.

## What It Does

This tool predicts materials properties (like band gap energy) just from knowing the chemical formula. Instead of running expensive simulations or lab experiments, you can get instant predictions for thousands of materials.

I built this to explore how well different ML approaches work for materials science problems - turns out Random Forest and XGBoost work surprisingly well for this.

## Results

Tested four different models on 1000 materials:

- **Random Forest**: Best for metals and semiconductors (MAE ~0.35 eV)
- **XGBoost**: Most consistent overall (R² ~0.90)
- **Neural Network**: Highest accuracy (R² ~0.91)
- **Ridge Regression**: Good baseline but struggles with non-linear patterns

### Example Predictions

| Material | Prediction | Actual | Notes |
|----------|------------|--------|-------|
| Silicon | 0.90 eV | 1.1 eV | Pretty close |
| Iron | 0.50 eV | 0.0 eV | Correctly identifies as metal |
| TiO2 | 2.61 eV | 3.0 eV | Good estimate |

## How to Use It

### Setup
```bash
git clone https://github.com/haruowo/materials-ml-predictor.git
cd materials-ml-predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key from [materialsproject.org](https://materialsproject.org) and set it:
```bash
export MP_API_KEY='your_key_here'
```

### Run the Pipeline
```bash
python src/data_collection.py      # Download materials data
python src/preprocessing.py        # Generate features
python src/training.py             # Train models
streamlit run web_app/app.py       # Launch web interface
```

## What I Learned

**The models work really well for certain materials but not others.** Random Forest nails predictions for metals (like Fe and Cu both predicted ~0.5 eV, actual 0 eV). But all the models struggle with insulators like NaCl.

Turns out the training data was super imbalanced - 79% metals, only 4% insulators. This is a common problem in materials databases since metals are easier to compute and more commonly studied.

**Potential fixes:**
- Sample materials more evenly across band gap ranges
- Use class weighting in the models
- Train separate models for different material types

## Tech Used

- PyTorch for neural networks
- scikit-learn and XGBoost for tree-based models
- pymatgen and matminer for materials features
- Streamlit for the web interface
- Materials Project API for data

## Project Structure
```
materials-ml-predictor/
├── src/
│   ├── data_collection.py    # Gets data from Materials Project
│   ├── preprocessing.py       # Creates features from formulas
│   └── training.py           # Trains and compares models
├── web_app/
│   └── app.py                # Interactive prediction interface
├── notebooks/
│   └── 01_data_exploration.ipynb
├── requirements.txt
└── README.md
```

## Future Ideas

- Add crystal structure features (currently only using composition)
- Try graph neural networks
- Predict multiple properties at once
- Add uncertainty estimates to predictions
- Balance the training dataset better

## Data Source

All materials data comes from the [Materials Project](https://materialsproject.org/), an open database of computed material properties.

## License

MIT License - feel free to use this however you want.

---

Built to learn more about ML for materials science. If you find bugs or have suggestions, open an issue!