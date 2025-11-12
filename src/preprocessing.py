"""
Feature Engineering for Materials ML

Converts chemical formulas and crystal structures into numerical features
suitable for machine learning models.
"""

import pandas as pd
import numpy as np
from pymatgen.core import Composition
from sklearn.preprocessing import StandardScaler
import joblib
import os
from tqdm import tqdm


class MaterialsFeatureEngineer:
    """Feature engineering for materials data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def create_simple_features(self, df, formula_column='formula'):
        """
        Create simpler, faster features for quick prototyping
        
        Args:
            df: DataFrame with chemical formulas
            formula_column: Name of column containing formulas
            
        Returns:
            DataFrame with added feature columns
        """
        print("\nCreating simple composition features...")
        
        features = []
        
        for idx, formula in tqdm(enumerate(df[formula_column]), total=len(df)):
            try:
                comp = Composition(formula)
                
                # Get element properties safely
                atomic_masses = []
                atomic_radii = []
                electronegativities = []
                groups = []
                rows = []
                
                for el in comp.elements:
                    atomic_masses.append(el.atomic_mass)
                    if el.atomic_radius is not None:
                        atomic_radii.append(el.atomic_radius)
                    if el.X is not None:
                        electronegativities.append(el.X)
                    groups.append(el.group)
                    rows.append(el.row)
                
                # Basic composition features
                feature_dict = {
                    'n_elements': len(comp.elements),
                    'mean_atomic_mass': np.mean(atomic_masses) if atomic_masses else 0,
                }
                
                # Atomic radius features
                if atomic_radii:
                    feature_dict['mean_atomic_radius'] = np.mean(atomic_radii)
                else:
                    feature_dict['mean_atomic_radius'] = 0
                
                # Electronegativity features
                if electronegativities:
                    feature_dict['mean_electronegativity'] = np.mean(electronegativities)
                    feature_dict['max_electronegativity'] = max(electronegativities)
                    feature_dict['min_electronegativity'] = min(electronegativities)
                    feature_dict['electronegativity_range'] = max(electronegativities) - min(electronegativities)
                else:
                    feature_dict['mean_electronegativity'] = 0
                    feature_dict['max_electronegativity'] = 0
                    feature_dict['min_electronegativity'] = 0
                    feature_dict['electronegativity_range'] = 0
                
                # Group and row features
                feature_dict['mean_group'] = np.mean(groups) if groups else 0
                feature_dict['mean_row'] = np.mean(rows) if rows else 0
                
                # Add fractional composition for most common elements
                for el in comp.elements:
                    feature_dict[f'frac_{el.symbol}'] = comp.get_atomic_fraction(el)
                
                features.append(feature_dict)
                
            except Exception as e:
                print(f"Error processing formula '{formula}': {e}")
                features.append({})
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(features)
        
        # Combine with original dataframe
        df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
        
        print(f"✓ Created {len(feature_df.columns)} feature columns")
        
        return df
    
    def prepare_features(self, df, target_column, feature_columns=None, test_size=0.2):
        """
        Prepare features for ML: select features, handle missing values, split data
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target variable column
            feature_columns: List of feature column names (if None, auto-select)
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        print("\nPreparing features for ML...")
        
        # Remove rows with missing target values
        df = df.dropna(subset=[target_column])
        print(f"  Dataset size after removing missing targets: {len(df)}")
        
        # Auto-select feature columns if not provided
        if feature_columns is None:
            # Exclude non-numeric and identifier columns
            exclude_cols = {target_column, 'material_id', 'formula', 'composition', 'elements'}
            feature_columns = []
            
            for col in df.columns:
                if col not in exclude_cols:
                    try:
                        # Check if column is numeric
                        col_dtype = df[col].dtype
                        if col_dtype.kind in ('i', 'f'):
                            feature_columns.append(col)
                    except Exception as e:
                        print(f"  Skipping column '{col}': {e}")
                        continue
        
        print(f"  Using {len(feature_columns)} features")
        
        # Handle missing values in features (impute with median)
        X = df[feature_columns].copy()
        
        # Fill NaN with median for each column
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        # Handle any remaining NaN or inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill any remaining NaN with 0
        X = X.fillna(0)
        
        y = df[target_column].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = feature_columns
        
        print(f"  Training set: {X_train_scaled.shape}")
        print(f"  Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns
    
    def save_preprocessor(self, save_dir='models'):
        """Save the scaler and feature names"""
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(self.feature_names, os.path.join(save_dir, 'feature_names.pkl'))
        print(f"✓ Saved preprocessor to {save_dir}")
    
    def load_preprocessor(self, save_dir='models'):
        """Load the scaler and feature names"""
        self.scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
        self.feature_names = joblib.load(os.path.join(save_dir, 'feature_names.pkl'))
        print(f"✓ Loaded preprocessor from {save_dir}")


def main():
    """Main function for feature engineering"""
    
    print("="*60)
    print("Feature Engineering")
    print("="*60)
    
    # Load raw data
    print("\nLoading band gap data...")
    df = pd.read_csv('data/band_gaps.csv')
    print(f"Loaded {len(df)} materials")
    
    # Initialize feature engineer
    engineer = MaterialsFeatureEngineer()
    
    # Create features (using simple method for speed)
    df_with_features = engineer.create_simple_features(df, formula_column='formula')
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    df_with_features.to_csv('data/processed/band_gaps_with_features.csv', index=False)
    print(f"\n✓ Saved processed data to data/processed/band_gaps_with_features.csv")
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test, feature_names = engineer.prepare_features(
        df_with_features, 
        target_column='band_gap'
    )
    
    # Save preprocessor
    engineer.save_preprocessor()
    
    # Save train/test splits
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    
    print("\n" + "="*60)
    print("Feature engineering complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train models: python src/training.py")
    print("2. Explore in notebooks: notebooks/02_model_training.ipynb")


if __name__ == "__main__":
    main()