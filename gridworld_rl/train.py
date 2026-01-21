"""
Main training script for GridWorld PPO agent.
ALIGNED with Q-learning for fair comparison.
"""
import os
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from datetime import datetime

from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

from configs.ppo_config import config
from models.networks import PolicyNetwork, ValueNetwork
from envs.gym_wrapper import make_gridworld_env


def main():
    print("=" * 70)
    print("GridWorld PPO Training - Aligned with Q-learning")
    print("=" * 70)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create vectorized environment using the correct method
    print(f"\nCreating {config.NUM_ENVS} parallel environments...")
    
    def make_env():
        """Factory function to create a single environment instance."""
        env = make_gridworld_env(
            size=config.ENV_SIZE,
            num_obstacles=config.NUM_OBSTACLES,
            goal_sequence=config.GOAL_SEQUENCE,
            max_steps=config.MAX_STEPS_PER_EPISODE
        )
        return env
    
    # Create vectorized environment (synchronous for GPU efficiency)
    env = SyncVectorEnv([make_env for _ in range(config.NUM_ENVS)])
    env = wrap_env(env)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"\nEnvironment Configuration:")
    print(f"  Grid size: {config.ENV_SIZE}x{config.ENV_SIZE}")
    print(f"  Obstacles: {config.NUM_OBSTACLES}")
    print(f"  Goal sequence: {config.GOAL_SEQUENCE}")
    print(f"  Max steps per episode: {config.MAX_STEPS_PER_EPISODE}")
    
    # Create models
    print("\nInitializing neural networks...")
    models = {
        "policy": PolicyNetwork(
            env.observation_space,
            env.action_space,
            device,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS
        ),
        "value": ValueNetwork(
            env.observation_space,
            env.action_space,
            device,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS
        )
    }
    
    # Print model info
    total_params_policy = sum(p.numel() for p in models["policy"].parameters())
    total_params_value = sum(p.numel() for p in models["value"].parameters())
    print(f"Policy network parameters: {total_params_policy:,}")
    print(f"Value network parameters: {total_params_value:,}")
    
    # Create memory
    memory = RandomMemory(
        memory_size=config.PPO_CONFIG["rollouts"],
        num_envs=config.NUM_ENVS,
        device=device
    )
    
    # Create PPO agent
    print("\nCreating PPO agent...")
    agent = PPO(
        models=models,
        memory=memory,
        cfg=config.PPO_CONFIG,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    
    # Configure trainer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(config.SAVE_DIR, f"{config.MODEL_NAME}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    cfg_trainer = {
        "timesteps": config.TOTAL_TIMESTEPS,
        "headless": True,
        "disable_progressbar": False,
        "close_environment_at_exit": True,
    }
    
    # Add TensorBoard logging
    if config.LOG_TENSORBOARD:
        cfg_trainer["experiment_name"] = config.MODEL_NAME
        cfg_trainer["experiment_directory"] = config.SAVE_DIR
    
    # Create trainer
    trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg_trainer)
    
    # Start training
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    print(f"Total timesteps: {config.TOTAL_TIMESTEPS:,}")
    print(f"Rollouts per iteration: {config.PPO_CONFIG['rollouts']}")
    print(f"Parallel environments: {config.NUM_ENVS}")
    print(f"Learning rate: {config.PPO_CONFIG['learning_rate']}")
    print(f"Discount factor (gamma): {config.PPO_CONFIG['discount_factor']}")
    print(f"Logs directory: {experiment_dir}")
    print("=" * 70 + "\n")
    
    # Train
    trainer.train()
    
    # Save final model
    model_path = os.path.join(experiment_dir, "final_agent.pt")
    agent.save(model_path)
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"\nTo view training logs:")
    print(f"  tensorboard --logdir {config.SAVE_DIR}")
    print(f"\nTo evaluate:")
    print(f"  python evaluate.py --model {model_path}")
    print(f"{'='*70}\n")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
