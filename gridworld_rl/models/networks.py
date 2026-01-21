"""
Neural network models for PPO agent.
Includes policy (actor) and value (critic) networks.
"""
import torch.nn as nn
from skrl.models.torch import Model, CategoricalMixin, DeterministicMixin


class PolicyNetwork(CategoricalMixin, Model):
    """
    Policy network (actor) for discrete action spaces.
    Outputs action probabilities for 4 actions: Up, Right, Down, Left.
    """
    
    def __init__(self, observation_space, action_space, device, 
                 hidden_size=128, num_layers=2):
        """
        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            device: torch device (cuda/cpu)
            hidden_size: Number of hidden units per layer
            num_layers: Number of hidden layers
        """
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        
        # Build network layers
        layers = []
        input_size = self.num_observations
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            ])
            input_size = hidden_size
        
        # Output layer (4 actions)
        layers.append(nn.Linear(hidden_size, self.num_actions))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights for stable training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def compute(self, inputs, role):
        """
        Forward pass.
        
        Args:
            inputs: Dictionary with 'states' key
            role: Role string (required by skrl interface)
            
        Returns:
            Tuple of (action_logits, empty_dict)
        """
        return self.net(inputs["states"]), {}


class ValueNetwork(DeterministicMixin, Model):
    """
    Value network (critic).
    Estimates state values for PPO's advantage calculation.
    """
    
    def __init__(self, observation_space, action_space, device,
                 hidden_size=128, num_layers=2):
        """
        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            device: torch device (cuda/cpu)
            hidden_size: Number of hidden units per layer
            num_layers: Number of hidden layers
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        
        # Build network layers
        layers = []
        input_size = self.num_observations
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            ])
            input_size = hidden_size
        
        # Output layer (single value)
        layers.append(nn.Linear(hidden_size, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights for stable training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def compute(self, inputs, role):
        """
        Forward pass.
        
        Args:
            inputs: Dictionary with 'states' key
            role: Role string (required by skrl interface)
            
        Returns:
            Tuple of (state_value, empty_dict)
        """
        return self.net(inputs["states"]), {}
