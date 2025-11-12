# How This All Works

Visual guide to the pipeline.

## The Big Picture
```
Chemical Formula → Features → ML Model → Band Gap Prediction
     "TiO2"     →  [98 numbers]  →  Random Forest  →  2.6 eV
```

## Detailed Pipeline

### Step 1: Get Data
```
Materials Project API
         ↓
Download 1000 materials
         ↓
Save to data/band_gaps.csv
```

What you get:
- Chemical formula (like "Si", "TiO2", "Fe2O3")
- Band gap (the thing we're predicting)
- Other properties (density, energy, etc.)

### Step 2: Create Features
```
Chemical Formula "TiO2"
         ↓
Parse with pymatgen
         ↓
Calculate 98 features:
  - n_elements = 2
  - mean_atomic_mass = 26.6
  - frac_Ti = 0.33
  - frac_O = 0.67
  - ... 94 more
         ↓
Save to data/processed/
```

### Step 3: Train Models
```
Split data: 80% train, 20% test
         ↓
Train 4 models in parallel:
  → Ridge Regression
  → Random Forest (100 trees)
  → XGBoost (200 boosting rounds)
  → Neural Network (256→128→64→1)
         ↓
Compare performance
         ↓
Save best models to models/
```

### Step 4: Make Predictions
```
New formula: "GaAs"
         ↓
Create features (same 98 as training)
         ↓
Run through all 4 models
         ↓
Show predictions:
  Ridge: 8.3 eV
  Random Forest: 1.4 eV  ← usually best
  XGBoost: 1.5 eV
  Neural Net: 1.3 eV
```

## File Flow
```
Input:
  Nothing! Just an API key

After data_collection.py:
  data/band_gaps.csv              (1000 materials, ~100 KB)

After preprocessing.py:
  data/processed/
    ├── band_gaps_with_features.csv
    ├── X_train.npy
    ├── X_test.npy
    ├── y_train.npy
    └── y_test.npy
  models/
    ├── scaler.pkl
    └── feature_names.pkl

After training.py:
  models/
    ├── ridge.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── neural_network.pth
  results/
    ├── model_comparison.json
    └── figures/
        ├── feature_importance.png
        └── training_curves.png
```

## How the Web App Works
```
User types "Si"
      ↓
Validate formula (pymatgen)
      ↓
Create 98 features
      ↓
Load all 4 trained models
      ↓
Get predictions from each
      ↓
Display results + graph
```

## Model Training Details

### Random Forest
```
For each of 100 trees:
  1. Take random sample of data
  2. At each split, consider random subset of features
  3. Split to minimize error
  4. Repeat until tree is deep enough
  
Prediction = average of all 100 trees
```

Why it works: Different trees learn different patterns, averaging reduces overfitting.

### XGBoost
```
Start with rough prediction (just the average)
For 200 rounds:
  1. Calculate errors
  2. Build small tree to predict those errors
  3. Add tree to ensemble with small weight
  4. Update predictions
  
Prediction = sum of all trees
```

Why it works: Each tree corrects mistakes of previous trees, gradient boosting optimizes this.

### Neural Network
```
Input (98 features)
      ↓
Dense layer (256 neurons) + ReLU
      ↓
Dropout (30%)
      ↓
Dense layer (128 neurons) + ReLU
      ↓
Dropout (30%)
      ↓
Dense layer (64 neurons) + ReLU
      ↓
Output (1 number = band gap)
```

Trained for 100 epochs with Adam optimizer.

## Feature Engineering Example

For "TiO2":
```python
comp = Composition("TiO2")

# Basic features
n_elements = 2
mean_atomic_mass = (Ti.atomic_mass + 2*O.atomic_mass) / 3

# Electronegativity
electronegativities = [1.54, 3.44, 3.44]  # Ti, O, O
mean_electronegativity = 2.81
range_electronegativity = 3.44 - 1.54 = 1.90

# Fractions
frac_Ti = 1/3 = 0.33
frac_O = 2/3 = 0.67

... 91 more features
```

## Time Breakdown

On my laptop:
```
data_collection.py:  ████░░░░░░  ~5 min  (depends on internet)
preprocessing.py:    ██░░░░░░░░  ~3 min  (CPU bound)
training.py:         ████████░░  ~15 min (CPU bound)
streamlit app:       instant     (just loading models)
```

Total time: ~25 minutes for full pipeline

## What Makes Good Predictions

**Works well when:**
- Material is similar to training data
- Band gap is 0-3 eV (most of training data)
- Common elements (not rare earth)

**Doesn't work well when:**
- Band gap > 5 eV (not enough training examples)
- Weird chemistry (noble gases, actinides)
- Needs crystal structure info (we only use composition)

---

That's the full pipeline! Pretty straightforward once you see it laid out.
