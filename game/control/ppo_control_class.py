"""
PPO Controller class for competing multiple models.

This module provides a class-based PPO controller that allows
instantiating multiple controllers with different model checkpoints.
Each controller maintains its own model and state independently.
"""

import numpy as np
import os
from .base_controller import BaseController

try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except ImportError:
    print("Warning: stable_baselines3 not available. Using rule-based control.")
    PPO_AVAILABLE = False


class PPOController(BaseController):
    """
    PPO Controller for autonomous car racing.
    
    Each instance loads and maintains its own PPO model,
    allowing multiple cars to use different model checkpoints.
    PPO models use discrete actions which are converted to continuous format.
    """
    
    def __init__(self, model_path="game/control/models/ppo_model", name=None):
        """
        Initialize the PPO controller with a specific model.
        
        Args:
            model_path: Path to the PPO model checkpoint file
            name: Optional name for this controller (for logging)
        """
        # Initialize base controller with name
        super().__init__(name=name or f"PPO_{os.path.basename(model_path)}")
        
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """
        Load the PPO model from the specified path.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not PPO_AVAILABLE:
            print(f"[{self.name}] PPO not available, using rule-based control")
            self.use_fallback = True
            return False
        
        if not os.path.exists(self.model_path):
            print(f"[{self.name}] Model file not found: {self.model_path}")
            print(f"[{self.name}] Falling back to rule-based control")
            self.use_fallback = True
            return False
        
        try:
            self.model = PPO.load(self.model_path, device="cpu")
            self.model_loaded = True
            print(f"[{self.name}] Successfully loaded PPO model from {self.model_path}")
            return True
        except Exception as e:
            print(f"[{self.name}] Failed to load PPO model: {e}")
            print(f"[{self.name}] Falling back to rule-based control")
            self.use_fallback = True
            return False
    
    def _discrete_to_continuous(self, discrete_action):
        """
        Convert discrete action to continuous action format.
        
        Args:
            discrete_action: integer (0-4) representing discrete action
                - 0: Do nothing (coast)
                - 1: Accelerate  
                - 2: Brake
                - 3: Turn left
                - 4: Turn right
        
        Returns:
            numpy array of shape (3,) containing [throttle, brake, steering]
        """
        # Ensure we have a scalar integer
        if isinstance(discrete_action, np.ndarray):
            discrete_action = int(discrete_action.item())
        else:
            discrete_action = int(discrete_action)
        
        # Map discrete actions to continuous values
        if discrete_action == 0:  # Do nothing (coast)
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif discrete_action == 1:  # Accelerate
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif discrete_action == 2:  # Brake
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif discrete_action == 3:  # Turn left
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        elif discrete_action == 4:  # Turn right
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            # Invalid action, default to coast
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def control(self, observation):
        """
        Calculate control actions based on observation.
        
        Uses the loaded PPO model if available, otherwise falls back
        to rule-based control logic. PPO models output discrete actions
        which are converted to continuous format.
        
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
            numpy array of shape (3,) containing [throttle, brake, steering]
            - throttle: 0.0 to 1.0
            - brake: 0.0 to 1.0
            - steering: -1.0 to 1.0
        """
        # Use PPO model if available
        if self.model_loaded and self.model is not None:
            try:
                # Use PPO model for prediction (deterministic=True for consistent racing)
                discrete_action, _ = self.model.predict(observation, deterministic=True)
                # Convert discrete action to continuous format
                return self._discrete_to_continuous(discrete_action)
            except Exception as e:
                print(f"[{self.name}] Error using PPO model: {e}")
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
        # Get base info and add PPO-specific fields
        info = super().get_info()
        info.update({
            'model_path': self.model_path,
            'model_loaded': self.model_loaded,
        })
        return info