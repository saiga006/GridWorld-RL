"""
Configuration file for PPO training - ALIGNED with Q-learning.
"""
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG


class Config:
    # Environment settings - MUST MATCH Q-learning
    ENV_SIZE = 10
    NUM_OBSTACLES = 5
    GOAL_SEQUENCE = [(2, 2), (5, 5), (8, 2), (1, 8)]  # Same as Q-learning!
    MAX_STEPS_PER_EPISODE = 200
    
    # Parallel environments for GPU efficiency
    NUM_ENVS = 16
    
    # Network architecture
    HIDDEN_SIZE = 256
    NUM_LAYERS = 3
    
    # PPO hyperparameters
    PPO_CONFIG = PPO_DEFAULT_CONFIG.copy()
    PPO_CONFIG.update({
        # Rollout settings
        "rollouts": 2048,
        "learning_epochs": 10,
        "mini_batches": 32,
        
        # Learning rates
        "learning_rate": 3e-4,
        "learning_rate_scheduler": None,
        
        # Discount and GAE - MATCH Q-learning gamma
        "discount_factor": 0.99,  # Same as Q-learning GAMMA
        "lambda": 0.95,
        
        # PPO clipping
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,
        
        # Loss coefficients
        "entropy_loss_scale": 0.02,
        "value_loss_scale": 0.5,
        
        # Optimization
        "grad_norm_clip": 0.5,
        "random_timesteps": 0,
        "learning_starts": 0,
        
        # Preprocessing
        "state_preprocessor": None,
        "value_preprocessor": None,
    })
    
    # Training settings
    # Q-learning: 3000 episodes × ~130 avg steps = ~390K steps
    # PPO with 16 envs: 400K timesteps for comparable training volume
    TOTAL_TIMESTEPS = 400000
    
    # Paths
    SAVE_DIR = "./runs"
    MODEL_NAME = "gridworld_ppo"
    
    # Device
    DEVICE = "cuda:0"
    
    # Logging
    LOG_TENSORBOARD = True
    
    # Evaluation settings - MATCH Q-learning
    EVAL_EPISODES = 100  # Same as Q-learning


config = Config()