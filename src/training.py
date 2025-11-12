"""
Model Training for Materials Property Prediction

Trains and evaluates multiple ML models:
- Linear Regression (baseline)
- Random Forest
- XGBoost
- Neural Network
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import json


class NeuralNetworkRegressor(nn.Module):
    """Feed-forward neural network for regression"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(NeuralNetworkRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


class ModelTrainer:
    """Train and evaluate multiple ML models"""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
    
    def train_linear_models(self):
        """Train linear baseline models"""
        print("\n" + "="*60)
        print("Training Linear Models (Baselines)")
        print("="*60)
        
        # Ridge Regression
        print("\n1. Ridge Regression...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(self.X_train, self.y_train)
        self.models['ridge'] = ridge
        self._evaluate_model(ridge, 'Ridge Regression')
        
        # Lasso Regression
        print("\n2. Lasso Regression...")
        lasso = Lasso(alpha=0.1)
        lasso.fit(self.X_train, self.y_train)
        self.models['lasso'] = lasso
        self._evaluate_model(lasso, 'Lasso Regression')
    
    def train_random_forest(self, n_estimators=100, max_depth=20):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("Training Random Forest")
        print("="*60)
        
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print(f"\nTraining with {n_estimators} trees...")
        rf.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf
        self._evaluate_model(rf, 'Random Forest')
        
        # Feature importance
        self._plot_feature_importance(rf, 'Random Forest')
    
    def train_xgboost(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("Training XGBoost")
        print("="*60)
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"\nTraining with learning_rate={learning_rate}...")
        xgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        self.models['xgboost'] = xgb_model
        self._evaluate_model(xgb_model, 'XGBoost')
        
        # Feature importance
        self._plot_feature_importance(xgb_model, 'XGBoost')
    
    def train_neural_network(self, hidden_dims=[256, 128, 64], epochs=100, batch_size=32, lr=0.001):
        """Train Neural Network model"""
        print("\n" + "="*60)
        print("Training Neural Network")
        print("="*60)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(self.X_train)
        y_train_tensor = torch.FloatTensor(self.y_train)
        X_test_tensor = torch.FloatTensor(self.X_test)
        y_test_tensor = torch.FloatTensor(self.y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = self.X_train.shape[1]
        model = NeuralNetworkRegressor(input_dim, hidden_dims=hidden_dims)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        print(f"\nTraining for {epochs} epochs...")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_test_tensor)
                val_loss = criterion(val_pred, y_test_tensor)
                val_losses.append(val_loss.item())
            
            train_losses.append(epoch_loss / len(train_loader))
            scheduler.step(val_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        
        # Save model
        self.models['neural_network'] = model
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train_tensor).numpy()
            y_pred_test = model(X_test_tensor).numpy()
        
        self._evaluate_predictions(y_pred_train, y_pred_test, 'Neural Network')
        
        # Plot training curve
        self._plot_training_curve(train_losses, val_losses)
    
    def _evaluate_model(self, model, model_name):
        """Evaluate a scikit-learn style model"""
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        self._evaluate_predictions(y_pred_train, y_pred_test, model_name)
    
    def _evaluate_predictions(self, y_pred_train, y_pred_test, model_name):
        """Calculate and print evaluation metrics"""
        # Training metrics
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        train_r2 = r2_score(self.y_train, y_pred_train)
        
        # Test metrics
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        self.results[model_name] = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2
        }
        
        print(f"\n{model_name} Results:")
        print(f"  Train MAE:  {train_mae:.4f} eV")
        print(f"  Train RMSE: {train_rmse:.4f} eV")
        print(f"  Train R²:   {train_r2:.4f}")
        print(f"  Test MAE:   {test_mae:.4f} eV")
        print(f"  Test RMSE:  {test_rmse:.4f} eV")
        print(f"  Test R²:    {test_r2:.4f}")
    
    def _plot_feature_importance(self, model, model_name):
        """Plot feature importance for tree-based models"""
        try:
            feature_names = joblib.load('models/feature_names.pkl')
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[-20:]  # Top 20
                
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(indices)), importances[indices])
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('Feature Importance')
                plt.title(f'{model_name} - Top 20 Important Features')
                plt.tight_layout()
                
                os.makedirs('results/figures', exist_ok=True)
                plt.savefig(f'results/figures/{model_name.lower().replace(" ", "_")}_importance.png', dpi=150)
                print(f"  ✓ Saved feature importance plot")
                plt.close()
        except Exception as e:
            print(f"  Could not plot feature importance: {e}")
    
    def _plot_training_curve(self, train_losses, val_losses):
        """Plot training curve for neural network"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Neural Network Training Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig('results/figures/nn_training_curve.png', dpi=150)
        print("  ✓ Saved training curve plot")
        plt.close()
    
    def save_models(self, save_dir='models'):
        """Save all trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'neural_network':
                torch.save(model.state_dict(), os.path.join(save_dir, 'neural_network.pth'))
            else:
                joblib.dump(model, os.path.join(save_dir, f'{name}.pkl'))
        
        print(f"\n✓ Saved {len(self.models)} models to {save_dir}")
    
    def save_results(self, save_path='results/model_comparison.json'):
        """Save evaluation results"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✓ Saved results to {save_path}")
    
    def print_comparison(self):
        """Print comparison of all models"""
        print("\n" + "="*60)
        print("Model Comparison Summary")
        print("="*60)
        
        df = pd.DataFrame(self.results).T
        df = df.round(4)
        print("\n", df.to_string())
        
        print("\n🏆 Best Model (by Test R²):")
        best_model = df['test_r2'].idxmax()
        print(f"  {best_model}: R² = {df.loc[best_model, 'test_r2']:.4f}")


def main():
    """Main training function"""
    
    print("="*60)
    print("Model Training")
    print("="*60)
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize trainer
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    
    # Train all models
    trainer.train_linear_models()
    trainer.train_random_forest(n_estimators=100, max_depth=20)
    trainer.train_xgboost(n_estimators=200, max_depth=8, learning_rate=0.1)
    trainer.train_neural_network(hidden_dims=[256, 128, 64], epochs=100, batch_size=64)
    
    # Compare results
    trainer.print_comparison()
    
    # Save everything
    trainer.save_models()
    trainer.save_results()
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Build web interface: streamlit run web_app/app.py")
    print("2. Analyze results in notebooks")


if __name__ == "__main__":
    main()