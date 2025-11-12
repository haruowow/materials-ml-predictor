"""
Data Collection from Materials Project API

This script downloads materials data including:
- Chemical compositions
- Crystal structures  
- Target properties (band gap, formation energy, etc.)
"""

import os
import json
import pandas as pd
from mp_api.client import MPRester
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MaterialsDataCollector:
    """Collect materials data from Materials Project"""
    
    def __init__(self, api_key=None):
        """
        Initialize the data collector
        
        Args:
            api_key: Materials Project API key. If None, reads from MP_API_KEY env variable
        """
        if api_key is None:
            api_key = os.environ.get('MP_API_KEY')
            if api_key is None:
                raise ValueError(
                    "No API key provided. Either pass api_key parameter or set MP_API_KEY environment variable.\n"
                    "Get your API key from: https://materialsproject.org/api"
                )
        
        self.mpr = MPRester(api_key)
        print("✓ Connected to Materials Project API")
    
    def collect_band_gap_data(self, max_materials=1000, save_path='data/band_gaps.csv'):
        """
        Collect materials with band gap data
        
        Args:
            max_materials: Maximum number of materials to collect
            save_path: Path to save the CSV file
        """
        print(f"\nCollecting band gap data for up to {max_materials} materials...")
        
        try:
            # Query materials with band gap data - simplified version
            docs = self.mpr.materials.summary.search(
                fields=[
                    "material_id",
                    "formula_pretty", 
                    "band_gap",
                    "formation_energy_per_atom",
                    "energy_per_atom",
                    "density",
                    "volume",
                    "nsites",
                    "elements",
                    "symmetry",
                    "is_stable"
                ],
                chunk_size=1000,
                num_chunks=max_materials // 1000 + 1
            )
            
            # Convert to list if it's a generator
            if not isinstance(docs, list):
                docs = list(docs)
            
            # Limit to max_materials
            docs = docs[:max_materials]
            
        except Exception as e:
            print(f"\nError with full query, trying simplified version...")
            print(f"Error details: {e}")
            
            # Fallback: simpler query
            docs = self.mpr.materials.summary.search(
                fields=["material_id", "formula_pretty", "band_gap"],
                chunk_size=1000
            )
            docs = list(docs)[:max_materials]
        
        # Convert to DataFrame
        data = []
        print("\nProcessing materials...")
        for doc in tqdm(docs):
            try:
                data.append({
                    'material_id': doc.material_id,
                    'formula': doc.formula_pretty,
                    'band_gap': doc.band_gap,
                    'formation_energy_per_atom': getattr(doc, 'formation_energy_per_atom', None),
                    'energy_per_atom': getattr(doc, 'energy_per_atom', None),
                    'density': getattr(doc, 'density', None),
                    'volume': getattr(doc, 'volume', None),
                    'nsites': getattr(doc, 'nsites', None),
                    'elements': ','.join(str(e) for e in doc.elements) if hasattr(doc, 'elements') else None,
                    'n_elements': len(doc.elements) if hasattr(doc, 'elements') else None,
                    'crystal_system': doc.symmetry.crystal_system.value if hasattr(doc, 'symmetry') and doc.symmetry else None,
                    'space_group': doc.symmetry.number if hasattr(doc, 'symmetry') and doc.symmetry else None,
                    'is_stable': getattr(doc, 'is_stable', None)
                })
            except Exception as e:
                print(f"Error processing material {doc.material_id}: {e}")
                continue
        
        df = pd.DataFrame(data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f"\n✓ Saved {len(df)} materials to {save_path}")
        
        # Print summary statistics
        print("\nDataset Summary:")
        print(f"  Total materials: {len(df)}")
        if 'band_gap' in df.columns and len(df) > 0:
            print(f"  Band gap range: {df['band_gap'].min():.2f} - {df['band_gap'].max():.2f} eV")
            print(f"  Mean band gap: {df['band_gap'].mean():.2f} eV")
            print(f"  Materials with band gap > 0: {(df['band_gap'] > 0).sum()}")
        
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            print(f"\n  Missing values:")
            for col, count in missing.items():
                print(f"    {col}: {count}")
        
        return df


def main():
    """Main function to run data collection"""
    
    print("="*60)
    print("Materials Project Data Collection")
    print("="*60)
    
    # Check for API key
    api_key = os.environ.get('MP_API_KEY')
    if not api_key:
        print("\n⚠ No API key found!")
        print("Please set your Materials Project API key:")
        print("  PowerShell: $env:MP_API_KEY='your_api_key_here'")
        print("  Command Prompt: set MP_API_KEY=your_api_key_here")
        print("\nGet your API key from: https://materialsproject.org/api")
        return
    
    # Initialize collector
    try:
        collector = MaterialsDataCollector(api_key)
    except Exception as e:
        print(f"\n❌ Error connecting to Materials Project: {e}")
        print("Please check your API key is correct.")
        return
    
    # Collect different datasets
    print("\n" + "="*60)
    print("1. Collecting Band Gap Data")
    print("="*60)
    
    # Start with smaller dataset to test
    try:
        band_gap_df = collector.collect_band_gap_data(max_materials=1000)
        
        print("\n" + "="*60)
        print("Data collection complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Explore the data in notebooks/01_data_exploration.ipynb")
        print("2. Run feature engineering: python src/preprocessing.py")
        print("3. Train models: python src/training.py")
        
    except Exception as e:
        print(f"\n❌ Error during data collection: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify your API key is valid")
        print("3. Try again - sometimes the API has temporary issues")
        return


if __name__ == "__main__":
    main()