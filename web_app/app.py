"""
Streamlit Web Application for Materials Property Prediction

Interactive interface for predicting material properties from chemical formulas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from pymatgen.core import Composition
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import MaterialsFeatureEngineer
from training import NeuralNetworkRegressor


# Page configuration
st.set_page_config(
    page_title="Materials Property Predictor",
    page_icon="🔬",
    layout="wide"
)


@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    try:
        models['ridge'] = joblib.load('models/ridge.pkl')
        models['random_forest'] = joblib.load('models/random_forest.pkl')
        models['xgboost'] = joblib.load('models/xgboost.pkl')
        
        # Load neural network
        feature_names = joblib.load('models/feature_names.pkl')
        input_dim = len(feature_names)
        nn_model = NeuralNetworkRegressor(input_dim)
        nn_model.load_state_dict(torch.load('models/neural_network.pth'))
        nn_model.eval()
        models['neural_network'] = nn_model
        
        return models, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, False


@st.cache_resource
def load_preprocessor():
    """Load feature engineering tools"""
    try:
        engineer = MaterialsFeatureEngineer()
        engineer.load_preprocessor('models')
        return engineer, True
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        return None, False


def predict_property(formula, models, engineer):
    """Make predictions for a given chemical formula"""
    
    try:
        # Create a temporary dataframe
        df = pd.DataFrame({'formula': [formula]})
        
        # Generate features
        df_features = engineer.create_simple_features(df, formula_column='formula')

        # Get the training feature names
        training_features = engineer.feature_names
        
        # Make sure all training features exist, fill missing with 0
        for feat in training_features:
            if feat not in df_features.columns:
                df_features[feat] = 0

        # Select only the features used during training (in correct order)
        X = df_features[training_features].fillna(0)
        
        # Scale features
        X_scaled = engineer.scaler.transform(X)
        
        # Make predictions with each model
        predictions = {}
        
        if 'ridge' in models:
            predictions['Ridge'] = models['ridge'].predict(X_scaled)[0]
        
        if 'random_forest' in models:
            predictions['Random Forest'] = models['random_forest'].predict(X_scaled)[0]
        
        if 'xgboost' in models:
            predictions['XGBoost'] = models['xgboost'].predict(X_scaled)[0]
        
        if 'neural_network' in models:
            X_tensor = torch.FloatTensor(X_scaled)
            with torch.no_grad():
                predictions['Neural Network'] = models['neural_network'](X_tensor).item()
        
        # Ensemble prediction (average)
        predictions['Ensemble (Average)'] = np.mean(list(predictions.values()))
        
        return predictions, df_features
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None
        st.error(traceback.format_exc())
        return None, None

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🔬 Materials Property Predictor")
    st.markdown("""
    Predict materials properties using machine learning! Enter a chemical formula below 
    to get predictions from multiple ML models.
    
    **Currently predicting:** Band gap (eV)
    """)
    
    # Load models
    with st.spinner("Loading models..."):
        models, models_loaded = load_models()
        engineer, engineer_loaded = load_preprocessor()
    
    if not models_loaded or not engineer_loaded:
        st.error("⚠️ Models not found. Please train models first by running: `python src/training.py`")
        st.stop()
    
    st.success(f"✓ Loaded {len(models)} models successfully!")
    
    # Sidebar with example formulas
    st.sidebar.header("📋 Example Formulas")
    st.sidebar.markdown("""
    Try these example materials:
    - **Si** - Silicon (semiconductor)
    - **GaAs** - Gallium Arsenide
    - **TiO2** - Titanium Dioxide
    - **CsPbI3** - Perovskite
    - **Fe2O3** - Iron Oxide
    - **NaCl** - Sodium Chloride
    - **MgB2** - Magnesium Diboride
    """)
    
    # Model information
    with st.sidebar.expander("ℹ️ About the Models"):
        st.markdown("""
        **Models used:**
        - Ridge Regression (baseline)
        - Random Forest
        - XGBoost
        - Neural Network
        - Ensemble (average)
        
        **Training data:** Materials Project database
        """)
    
    # Main input section
    st.header("🧪 Make a Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        formula_input = st.text_input(
            "Enter Chemical Formula",
            value="Si",
            help="Enter a valid chemical formula (e.g., Fe2O3, NaCl, TiO2)"
        )
    
    with col2:
        predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    # Make prediction when button is clicked
    if predict_button or formula_input:
        
        # Validate formula
        try:
            comp = Composition(formula_input)
            st.success(f"✓ Valid formula: {comp.reduced_formula}")
        except Exception as e:
            st.error(f"❌ Invalid formula: {e}")
            st.stop()
        
        # Make predictions
        with st.spinner("Making predictions..."):
            predictions, features = predict_property(formula_input, models, engineer)
        
        if predictions is not None:
            st.header("📊 Prediction Results")
            
            # Display predictions in columns
            cols = st.columns(len(predictions))
            
            for i, (model_name, pred_value) in enumerate(predictions.items()):
                with cols[i]:
                    st.metric(
                        label=model_name,
                        value=f"{pred_value:.3f} eV",
                        delta=None
                    )
            
            # Visualization
            st.subheader("Model Comparison")
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(predictions.keys()),
                    y=list(predictions.values()),
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                )
            ])
            
            fig.update_layout(
                title="Band Gap Predictions by Model",
                xaxis_title="Model",
                yaxis_title="Band Gap (eV)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Material composition details
            with st.expander("📝 Material Composition Details"):
                comp_data = {
                    'Element': [str(el) for el in comp.elements],
                    'Atomic Fraction': [comp.get_atomic_fraction(el) for el in comp.elements],
                    'Weight Fraction': [comp.get_wt_fraction(el) for el in comp.elements],
                }
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
            
            # Feature values (sample)
            with st.expander("🔍 Feature Values (Sample)"):
                # Show first 20 features
                sample_features = features[engineer.feature_names[:20]].iloc[0]
                feature_df = pd.DataFrame({
                    'Feature': sample_features.index,
                    'Value': sample_features.values
                })
                st.dataframe(feature_df, use_container_width=True, height=400)
    
    # Batch prediction section
    st.header("📦 Batch Prediction")
    st.markdown("Upload a CSV file with a 'formula' column to predict properties for multiple materials.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            if 'formula' not in df_batch.columns:
                st.error("CSV must contain a 'formula' column")
                st.stop()
            
            st.write(f"Loaded {len(df_batch)} formulas")
            st.dataframe(df_batch.head())
            
            if st.button("🚀 Run Batch Prediction"):
                with st.spinner("Processing batch predictions..."):
                    results = []
                    
                    progress_bar = st.progress(0)
                    
                    for idx, formula in enumerate(df_batch['formula']):
                        try:
                            preds, _ = predict_property(formula, models, engineer)
                            if preds:
                                result = {'formula': formula}
                                result.update(preds)
                                results.append(result)
                        except:
                            pass
                        
                        progress_bar.progress((idx + 1) / len(df_batch))
                    
                    results_df = pd.DataFrame(results)
                    
                    st.success(f"✓ Completed predictions for {len(results_df)} materials")
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit | Models trained on Materials Project data</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()