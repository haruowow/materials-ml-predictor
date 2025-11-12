# Getting Started

Quick guide to understanding and running this project.

## What's This Project About?

I wanted to see if machine learning could predict materials properties without running expensive simulations. Turns out it works pretty well! The models can predict band gaps for most semiconductors and metals with decent accuracy.

## What's Inside
```
materials-ml-predictor/
├── src/                  # Main code
│   ├── data_collection.py
│   ├── preprocessing.py
│   └── training.py
├── web_app/             # Streamlit interface
│   └── app.py
├── notebooks/           # Jupyter analysis
└── README.md
```

## Quick Start
```bash
# 1. Install everything
pip install -r requirements.txt

# 2. Get API key from materialsproject.org
export MP_API_KEY='your_key'

# 3. Run the pipeline
python src/data_collection.py
python src/preprocessing.py
python src/training.py

# 4. Try the web app
streamlit run web_app/app.py
```

## What Each File Does

**data_collection.py** - Downloads materials data from Materials Project. Takes about 5-10 minutes for 1000 materials.

**preprocessing.py** - Turns chemical formulas into numbers that ML models can understand. Creates ~98 features per material.

**training.py** - Trains 4 different models and compares them. Takes 10-20 minutes depending on your computer.

**app.py** - Web interface where you can type in any chemical formula and get predictions from all models.

## Things I Found Out

- Random Forest works really well for this (better than I expected)
- The training data is super imbalanced - 79% metals!
- Models struggle with insulators because there aren't many in the dataset
- Neural networks are overkill for this problem honestly

## If Something Breaks

**"No API key found"**
```bash
export MP_API_KEY='your_key_here'
```

**Models predict weird values**
- Probably forgot to run preprocessing first
- Make sure you have the data/ folder with band_gaps.csv

**Web app crashes**
- Check that models/ folder exists with the .pkl files
- Retrain if needed: `python src/training.py`

## What I'd Change

If I were to redo this:
1. Collect more balanced data (equal amounts of metals, semiconductors, insulators)
2. Add crystal structure features, not just composition
3. Try graph neural networks
4. Add confidence intervals to predictions

## Playing Around

The web app is the fun part. Try:
- **Si** (silicon) - should predict ~1 eV
- **Fe** (iron) - should predict ~0 eV
- **NaCl** (salt) - this one fails (predicts way too low)

The failure cases are actually the interesting part - they show the model's limitations.

---

Questions? Open an issue or fork it and experiment!
