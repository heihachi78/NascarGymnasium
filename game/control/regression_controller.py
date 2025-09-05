"""
Regression-based controller for car racing.

This module implements controllers that use various regression models
to learn direct mappings from observations to control actions.
"""

import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available. RegressionController will only use fallback control.")

from .base_controller import BaseController


class RegressionController(BaseController):
    """
    Controller that uses regression models for direct observation-to-action mapping.
    
    Supports multiple regression model types:
    - Linear Regression
    - Ridge Regression  
    - Random Forest
    - Neural Network (MLP)
    """
    
    def __init__(self, name=None, model_type='random_forest', model_path=None):
        """
        Initialize the regression controller.
        
        Args:
            name: Optional name for this controller
            model_type: Type of regression model ('linear', 'ridge', 'random_forest', 'neural_network')
            model_path: Path to saved model file (if None, creates untrained model)
        """
        super().__init__(name or f"RegressionController_{model_type}")
        
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # Model hyperparameters
        self.model_params = {
            'linear': {},
            'ridge': {'alpha': 1.0},
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            },
            'neural_network': {
                'hidden_layer_sizes': (128, 64, 32),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'batch_size': 'auto',
                'learning_rate': 'constant',
                'learning_rate_init': 0.001,
                'max_iter': 1000,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 10
            }
        }
        
        # Initialize model and scaler
        self._initialize_model()
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _initialize_model(self):
        """Initialize the regression model based on model_type."""
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn not available. Model will use fallback control only.")
            self.model = None
            self.scaler = None
            return
            
        params = self.model_params.get(self.model_type, {})
        
        if self.model_type == 'linear':
            self.model = LinearRegression(**params)
        elif self.model_type == 'ridge':
            self.model = Ridge(**params)
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**params)
        elif self.model_type == 'neural_network':
            self.model = MLPRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Always use a scaler for consistent input normalization
        self.scaler = StandardScaler()
    
    def control(self, observation):
        """
        Generate control actions using the regression model.
        
        Args:
            observation: numpy array of shape (38,) containing car state
        
        Returns:
            numpy array of shape (2,) containing [throttle_brake, steering]
        """
        if not self.is_trained or self.model is None:
            # Fall back to rule-based control if model not trained
            self.use_fallback = True
            return self._fallback_control(observation)
        
        try:
            # Ensure observation is the right shape and type
            obs = np.array(observation, dtype=np.float32).reshape(1, -1)
            
            # Check for valid observation size
            if obs.shape[1] != 38:
                raise ValueError(f"Expected 38 features, got {obs.shape[1]}")
            
            # Scale input features
            obs_scaled = self.scaler.transform(obs)
            
            # Predict actions
            action_pred = self.model.predict(obs_scaled)[0]
            
            # Ensure action is the right shape and within bounds
            action = np.array(action_pred, dtype=np.float32)
            if len(action) != 2:
                raise ValueError(f"Expected 2 actions, got {len(action)}")
            
            # Clip actions to valid range
            action[0] = np.clip(action[0], -1.0, 1.0)  # throttle_brake
            action[1] = np.clip(action[1], -1.0, 1.0)  # steering
            
            self.use_fallback = False
            return action
            
        except Exception:
            # Fall back to rule-based control on any error
            self.use_fallback = True
            return self._fallback_control(observation)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the regression model.
        
        Args:
            X_train: Training observations (n_samples, 38)
            y_train: Training actions (n_samples, 2)
            X_val: Validation observations (optional)
            y_val: Validation actions (optional)
        """
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        
        print(f"🤖 Training {self.model_type} model...")
        print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Fit scaler on training data
        self.scaler.fit(X_train)
        
        # Scale training data
        X_train_scaled = self.scaler.transform(X_train)
        
        # Train model
        try:
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            print(f"✅ Model training completed")
            
            # Evaluate on training data
            train_score = self.model.score(X_train_scaled, y_train)
            print(f"Training R² score: {train_score:.4f}")
            
            # Evaluate on validation data if provided
            if X_val is not None and y_val is not None:
                X_val = np.array(X_val, dtype=np.float32)
                y_val = np.array(y_val, dtype=np.float32)
                X_val_scaled = self.scaler.transform(X_val)
                val_score = self.model.score(X_val_scaled, y_val)
                print(f"Validation R² score: {val_score:.4f}")
                
                # Check for overfitting
                if train_score - val_score > 0.2:
                    print("⚠️  Possible overfitting detected (train-val score gap > 0.2)")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            self.is_trained = False
            return False
    
    def save_model(self, model_path: str):
        """Save trained model and scaler to file."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'model_params': self.model_params[self.model_type],
            'is_trained': self.is_trained
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model and scaler from file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data.get('model_type', self.model_type)
            self.is_trained = model_data.get('is_trained', True)
            
            print(f"📂 Model loaded from {model_path}")
            print(f"Model type: {self.model_type}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.is_trained = False
            return False
    
    def predict_batch(self, observations):
        """
        Predict actions for a batch of observations.
        
        Args:
            observations: numpy array of shape (n_samples, 38)
            
        Returns:
            numpy array of shape (n_samples, 2) containing predicted actions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        observations = np.array(observations, dtype=np.float32)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        
        # Scale observations
        obs_scaled = self.scaler.transform(observations)
        
        # Predict actions
        actions_pred = self.model.predict(obs_scaled)
        
        # Clip to valid range
        actions_pred[:, 0] = np.clip(actions_pred[:, 0], -1.0, 1.0)  # throttle_brake
        actions_pred[:, 1] = np.clip(actions_pred[:, 1], -1.0, 1.0)  # steering
        
        return actions_pred.astype(np.float32)
    
    def get_feature_importance(self):
        """Get feature importance for tree-based models."""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, return absolute coefficients as importance
            coef = self.model.coef_
            if coef.ndim > 1:
                # Multi-output case, average across outputs
                return np.mean(np.abs(coef), axis=0)
            else:
                return np.abs(coef)
        else:
            return None
    
    def get_info(self):
        """Get information about this controller."""
        info = super().get_info()
        info.update({
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'model_path': self.model_path,
            'has_scaler': self.scaler is not None
        })
        
        if self.is_trained and hasattr(self.model, 'n_features_in_'):
            info['n_features'] = self.model.n_features_in_
            
        return info
    
    def evaluate_predictions(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test observations (n_samples, 38)
            y_test: True test actions (n_samples, 2)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)
        
        # Make predictions
        y_pred = self.predict_batch(X_test)
        
        # Calculate metrics
        if SKLEARN_AVAILABLE:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        else:
            return {'error': 'scikit-learn not available'}
        
        # Overall metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Per-action metrics
        mse_throttle = mean_squared_error(y_test[:, 0], y_pred[:, 0])
        mse_steering = mean_squared_error(y_test[:, 1], y_pred[:, 1])
        
        mae_throttle = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
        mae_steering = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
        
        r2_throttle = r2_score(y_test[:, 0], y_pred[:, 0])
        r2_steering = r2_score(y_test[:, 1], y_pred[:, 1])
        
        return {
            'mse_overall': mse,
            'mae_overall': mae,
            'r2_overall': r2,
            'mse_throttle': mse_throttle,
            'mse_steering': mse_steering,
            'mae_throttle': mae_throttle,
            'mae_steering': mae_steering,
            'r2_throttle': r2_throttle,
            'r2_steering': r2_steering,
            'n_samples': len(y_test)
        }