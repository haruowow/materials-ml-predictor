# Project Deep Dive

Technical details about how this works and what I learned.

## The Problem

Discovering new materials is slow. You either:
1. Make it in a lab ($$$, weeks)
2. Simulate it with DFT (hours per material)
3. Use ML to screen thousands instantly

I wanted to see how well option 3 actually works.

## How It Works

### 1. Data Collection
Downloaded 1000 materials from Materials Project using their API. Each material has:
- Chemical formula (like "TiO2")
- Band gap (the property we're predicting)
- Other stuff (density, energy, crystal structure)

### 2. Feature Engineering
Turned chemical formulas into ~98 numbers:
- Number of elements
- Average atomic mass
- Electronegativity stats
- Atomic radius stats
- Fraction of each element

Example: "TiO2" becomes:
```
n_elements: 2
mean_atomic_mass: 26.6
mean_electronegativity: 2.76
frac_Ti: 0.33
frac_O: 0.67
... 93 more features
```

### 3. Model Training
Trained 4 different models to compare:

**Ridge Regression** (baseline)
- Simple linear model
- Fast to train (seconds)
- Predicts way too high for everything
- **Verdict**: Not good enough

**Random Forest** (surprisingly good)
- 100 decision trees
- Takes 2-5 minutes to train
- Excellent for metals and semiconductors
- **Verdict**: Best practical choice

**XGBoost** (industry standard)
- Gradient boosted trees
- Takes 5-7 minutes
- Slightly better than Random Forest
- **Verdict**: Also great

**Neural Network** (overkill?)
- 4 layers: 256→128→64→1
- Takes 10-15 minutes
- Highest R² score
- **Verdict**: Best accuracy but not worth the complexity

## Results

Tested on 200 materials (20% holdout):

| Model | R² Score | MAE (eV) | What It's Good At |
|-------|----------|----------|-------------------|
| Ridge | 0.83 | 0.48 | Nothing really |
| Random Forest | 0.89 | 0.35 | Metals, semiconductors |
| XGBoost | 0.90 | 0.32 | Everything except insulators |
| Neural Net | 0.91 | 0.30 | Same as XGBoost |

## Real-World Testing

I tested the models on materials I know the answer to:

**Silicon (semiconductor, 1.1 eV actual)**
- Random Forest: 0.90 eV ✓
- Neural Net: 1.50 eV ✓
- Ridge: 10.53 eV ✗

**Iron (metal, 0 eV actual)**
- Random Forest: 0.50 eV ✓
- XGBoost: 1.65 eV ~
- Ridge: 11.52 eV ✗

**Sodium Chloride (insulator, 8.5 eV actual)**
- Random Forest: 0.68 eV ✗
- XGBoost: 1.39 eV ✗
- Ridge: 7.12 eV ~ (accidentally closest)

## The Big Problem I Found

**Data imbalance is killing the insulator predictions.**

Training data breakdown:
- 787 metals (79%)
- 154 semiconductors (15%)
- 41 insulators (4%)
- 18 mid-range (2%)

No wonder the models think everything is a metal or semiconductor!

## What I'd Do Differently

### Short-term fixes:
1. **Stratified sampling** - collect equal amounts of each type
2. **Class weights** - tell the model insulators are important
3. **Separate models** - one for each band gap range

### Long-term ideas:
1. **Crystal structure features** - currently only using composition
2. **Graph neural networks** - treat atoms as graph nodes
3. **Transfer learning** - start with a model trained on millions of materials
4. **Uncertainty quantification** - tell me when the model isn't sure

## Tech Stack

**Languages & Core:**
- Python 3.13
- NumPy, pandas for data

**ML Libraries:**
- scikit-learn (Random Forest, Ridge)
- XGBoost
- PyTorch (Neural Network)

**Materials Science:**
- pymatgen (parse chemical formulas)
- matminer (generate features)
- Materials Project API (data source)

**Interface:**
- Streamlit (web app)
- Plotly (interactive graphs)

## Performance Notes

On my laptop (specs here if you want):
- Data collection: ~2 minutes for 1000 materials
- Feature engineering: ~3 minutes
- Random Forest training: ~3 minutes
- Full pipeline: ~20 minutes total

## Interesting Findings

1. **More complex ≠ better** - Random Forest performs almost as well as the Neural Network but trains way faster

2. **Linear models fail hard** - Ridge regression is consistently terrible because band gap has really non-linear relationships

3. **Materials databases have bias** - They contain way more metals because they're easier to compute

4. **Feature engineering matters** - Simple composition features work surprisingly well; didn't even need crystal structure

## What This Is Useful For

**What it's good at:**
- Quickly screening semiconductors for electronics
- Identifying metals (always predicts ~0 eV)
- Getting ballpark estimates for materials discovery

**What it's not good at:**
- Insulators (predicts way too low)
- Materials with rare elements (not in training data)
- Precise predictions (±0.3 eV error is pretty big)

**Realistic use case:**
Screen 10,000 candidate materials → narrow to 100 with ML → run DFT on those 100 → pick best 10 for lab testing

## Code Quality Notes

Things I'm happy with:
- Modular code (each script does one thing)
- Error handling for edge cases
- Documentation in the code
- Web interface is actually usable

Things I'd improve:
- Add unit tests
- Better logging
- Config file instead of hardcoded parameters
- CLI arguments instead of editing source

---

Overall, this was a fun project and I learned a lot about both ML and materials science. The models work surprisingly well given how simple the features are.