"""
A2C Controller class for competing multiple models.

This module provides a class-based A2C controller that allows
instantiating multiple controllers with different model checkpoints.
Each controller maintains its own model and state independently.
"""

import numpy as np
import os
from .base_controller import BaseController

try:
    from stable_baselines3 import A2C
    A2C_AVAILABLE = True
except ImportError:
    print("Warning: stable_baselines3 not available. Using rule-based control.")
    A2C_AVAILABLE = False


class A2CController(BaseController):
    """
    A2C Controller for autonomous car racing.
    
    Each instance loads and maintains its own A2C model,
    allowing multiple cars to use different model checkpoints.
    A2C is an off-policy algorithm that works with continuous actions.
    """
    
    def __init__(self, model_path="game/control/models/a2c_model.zip", name=None):
        """
        Initialize the A2C controller with a specific model.
        
        Args:
            model_path: Path to the A2C model checkpoint file
            name: Optional name for this controller (for logging)
        """
        # Initialize base controller with name
        super().__init__(name=name or f"A2C_{os.path.basename(model_path)}")
        
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """
        Load the A2C model from the specified path.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not A2C_AVAILABLE:
            print(f"[{self.name}] A2C not available, using rule-based control")
            self.use_fallback = True
            return False
        
        if not os.path.exists(self.model_path):
            print(f"[{self.name}] Model file not found: {self.model_path}")
            print(f"[{self.name}] Falling back to rule-based control")
            self.use_fallback = True
            return False
        
        try:
            self.model = A2C.load(self.model_path, device='cpu')
            self.model_loaded = True
            print(f"[{self.name}] Successfully loaded A2C model from {self.model_path}")
            return True
        except Exception as e:
            print(f"[{self.name}] Failed to load A2C model: {e}")
            print(f"[{self.name}] Falling back to rule-based control")
            self.use_fallback = True
            return False
    
    def control(self, observation):
        """
        Calculate control actions based on observation.
        
        Uses the loaded A2C model if available, otherwise falls back
        to rule-based control logic.
        
        Args:
            observation: numpy array of shape (29,) containing:
                - Position (x, y): indices 0-1
                - Velocity (x, y, magnitude): indices 2-4
                - Orientation and angular velocity: indices 5-6
                - Tire loads (4): indices 7-10
                - Tire temperatures (4): indices 11-14
                - Tire wear (4): indices 15-18
                - Collision data (impulse, angle): indices 19-20
                - Distance sensor readings (8 directions): indices 21-28
        
        Returns:
            numpy array of shape (2,) containing [throttle_brake, steering]
            - throttle_brake: -1.0 (full brake) to 1.0 (full throttle)
            - steering: -1.0 to 1.0
        """
        # Use A2C model if available
        if self.model_loaded and self.model is not None:
            try:
                # Use A2C model for prediction (deterministic=True for consistent racing)
                # NOTE: Existing models trained with 3-element actions will not work correctly
                # Models need to be retrained with 2-element action space
                action, _ = self.model.predict(observation, deterministic=True)
                return action.astype(np.float32)
            except Exception as e:
                print(f"[{self.name}] Error using A2C model: {e}")
                print(f"[{self.name}] Switching to fallback control")
                self.use_fallback = True
        
        # Fallback to rule-based control
        return self._fallback_control(observation)
    
    def get_info(self):
        """
        Get information about this controller.
        
        Returns:
            dict: Controller information including model path and status
        """
        # Get base info and add A2C-specific fields
        info = super().get_info()
        info.update({
            'model_path': self.model_path,
            'model_loaded': self.model_loaded,
        })
        return info