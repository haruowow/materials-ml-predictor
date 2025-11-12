# Quick Setup Guide

Get this running on your machine in about 30 minutes.

## What You Need

- Python 3.8 or newer
- A Materials Project account (free)
- About 2GB of disk space

## Step 1: Download the Code
```bash
git clone https://github.com/haruowo/materials-ml-predictor.git
cd materials-ml-predictor
```

## Step 2: Set Up Python Environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This takes 5-10 minutes. Go grab coffee ☕

## Step 3: Get Your API Key

1. Go to https://materialsproject.org
2. Sign up (it's free)
3. Go to Dashboard → API
4. Copy your key

Then set it:
```bash
export MP_API_KEY='paste_your_key_here'
```

## Step 4: Run Everything
```bash
# Download data (~5 minutes)
python src/data_collection.py

# Process features (~2 minutes)
python src/preprocessing.py

# Train models (~15 minutes)
python src/training.py

# Launch web app
streamlit run web_app/app.py
```

Your browser should open automatically to localhost:8501

## If You Just Want to See It Work

Skip training and use my pre-trained models (if I included them). Just run:
```bash
streamlit run web_app/app.py
```

## Common Issues

**Python not found**
- Install from python.org
- Make sure "Add to PATH" is checked during install

**pip install fails**
- Try: `pip install --upgrade pip`
- Or use conda: `conda install -c conda-forge pymatgen`

**API key doesn't work**
- Make sure you copied the whole thing
- Check if it has quotes: `export MP_API_KEY='key'` not `export MP_API_KEY=key`

**Training takes forever**
- Normal on older computers
- You can reduce the dataset size in data_collection.py (change `max_materials=1000` to like 200)

## What You'll Get

After running everything:
- `data/` folder with 1000 materials
- `models/` folder with 4 trained models
- A web interface to predict any material

## Next Steps

Try predicting some materials:
- Type "Si" → should get ~1 eV
- Type "GaAs" → should get ~1.4 eV
- Type "NaCl" → will fail (models don't handle insulators well)

Check out the Jupyter notebook to explore the data:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

That's it! Pretty straightforward once everything's installed.