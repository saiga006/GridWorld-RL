"""
Evaluation script for PPO agent - ALIGNED with Q-learning evaluation.
Computes exact same metrics: success_rate, collision_rate, SPL, avg_steps_success
"""
import torch
import numpy as np
import csv
import os

from configs.ppo_config import config
from models.networks import PolicyNetwork, ValueNetwork
from envs.gym_wrapper import GridWorldGym
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env
from utils.metrics import calculate_optimal_path_length_manhattan, calculate_spl_manhattan
import argparse
import time

def evaluate_agent(agent, eval_episodes=100, verbose=True, render=False):
    """
    Evaluate agent with metrics MATCHING Q-learning exactly.
    
    Args:
        agent: Trained PPO agent
        eval_episodes: Number of evaluation episodes
        verbose: Print detailed progress
        
    Returns:
        dict: Evaluation metrics matching Q-learning format
    """
    # Create evaluation environment - SAME CONFIG as Q-learning
    env = GridWorldGym(
        size=config.ENV_SIZE,
        num_obstacles=config.NUM_OBSTACLES,
        goal_sequence=config.GOAL_SEQUENCE,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        render_mode='human' if render else None  # Enable rendering
    )
    env = wrap_env(env)
    
    # Calculate optimal path length using Manhattan (same as Q-learning)
    optimal_length = calculate_optimal_path_length_manhattan((0, 0), config.GOAL_SEQUENCE)
    
    if verbose:
        print(f"\nOptimal Path Length for sequence: {optimal_length}")
        print(f"Goal sequence: {config.GOAL_SEQUENCE}")
    
    # Metrics tracking
    total_successes = 0
    total_collisions = 0
    spl_accum = 0
    steps_success_accum = 0
    
    # Logging - SAME format as Q-learning
    log_filename = "ppo_eval_log.csv"
    with open(log_filename, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'success', 'collision', 'steps', 'spl']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    if verbose:
        print(f"\nStarting evaluation for {eval_episodes} episodes...")
        print("-" * 70)
    
    for episode in range(1, eval_episodes + 1):
        states, _ = env.reset()
        done = False
        steps = 0
        collision_occurred = False
        
        while not done and steps < config.MAX_STEPS_PER_EPISODE:
            if render:
                env.render()  # Show current state
                time.sleep(0.1)  # Slow down to see movements
            
            # Greedy action (no exploration during evaluation)
            with torch.no_grad():
                actions = agent.act(states, timestep=0, timesteps=0)[0]
            
            states, rewards, terminated, truncated, infos = env.step(actions)
            
            # Check collision - SAME logic as Q-learning (reward == -10)
            if rewards[0].item() <= -10:
                collision_occurred = True
            
            done = terminated[0] or truncated[0]
            steps += 1
        
        # Success check - SAME as Q-learning
        # Q-learning: is_success = done and steps < MAX_STEPS
        is_success = terminated[0].item() and steps < config.MAX_STEPS_PER_EPISODE
        
        # Calculate SPL - SAME formula as Q-learning
        if is_success:
            total_successes += 1
            steps_success_accum += steps
            # Q-learning formula: 1.0 * optimal_length / max(steps, optimal_length)
            spl = calculate_spl_manhattan(True, steps, optimal_length)
        else:
            spl = 0.0
        
        if collision_occurred:
            total_collisions += 1
        
        spl_accum += spl
        
        # Log episode - SAME format as Q-learning
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'episode': episode,
                'success': 1 if is_success else 0,
                'collision': 1 if collision_occurred else 0,
                'steps': steps,
                'spl': spl
            })
        
        if verbose and episode % 10 == 0:
            print(f"Episode {episode:3d}: Steps={steps:3d}, Success={int(is_success)}, "
                  f"Collision={int(collision_occurred)}, SPL={spl:.3f}")
    
    env.close()
    
    # Calculate final metrics - SAME as Q-learning
    success_rate = total_successes / eval_episodes
    collision_rate = total_collisions / eval_episodes
    avg_spl = spl_accum / eval_episodes
    avg_steps_success = (steps_success_accum / total_successes) if total_successes > 0 else 0
    
    if verbose:
        # Print results - SAME format as Q-learning
        print("-" * 70)
        print(f"Evaluation Results ({eval_episodes} Episodes)")
        print(f"Success Rate: {success_rate:.2f}")
        print(f"Collision Rate: {collision_rate:.2f}")
        print(f"SPL: {avg_spl:.2f}")
        print(f"Avg Steps (Success): {avg_steps_success:.2f}")
        print("-" * 70)
    
    # Save summary - SAME format as Q-learning
    with open("ppo_eval_summary.txt", "w") as f:
        f.write(f"success_rate,{success_rate}\n")
        f.write(f"collision_rate,{collision_rate}\n")
        f.write(f"SPL,{avg_spl}\n")
        f.write(f"avg_steps_success,{avg_steps_success}\n")
    
    print(f"\nEvaluation data saved to {log_filename}")
    print(f"Summary saved to ppo_eval_summary.txt")
    
    return {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'SPL': avg_spl,
        'avg_steps_success': avg_steps_success,
        'total_successes': total_successes
    }


def main():
    
    
    parser = argparse.ArgumentParser(description='Evaluate PPO agent - Aligned with Q-learning')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to saved model checkpoint (e.g., runs/gridworld_ppo_*/final_agent.pt)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100, same as Q-learning)')
    parser.add_argument('--render', action='store_true',  
                        help='Render environment during evaluation')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating model: {args.model}")
    
    # Create environment
    env = GridWorldGym(
        size=config.ENV_SIZE,
        num_obstacles=config.NUM_OBSTACLES,
        goal_sequence=config.GOAL_SEQUENCE,
        max_steps=config.MAX_STEPS_PER_EPISODE
    )
    env = wrap_env(env)
    
    # Create models
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
    
    # Create agent
    memory = RandomMemory(memory_size=16, num_envs=1, device=device)
    agent = PPO(
        models=models,
        memory=memory,
        cfg=config.PPO_CONFIG,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    
    # Load trained weights
    print(f"Loading model...")
    agent.load(args.model)
    agent.set_running_mode("eval")
    
    env.close()
    
    # Evaluate
    results = evaluate_agent(agent, eval_episodes=args.episodes,verbose=True, render=args.render)
    
    print(f"\nComparison with Q-learning:")
    print(f"  - Both use same environment config")
    print(f"  - Both use Manhattan distance for SPL")
    print(f"  - Both use same metrics: success_rate, collision_rate, SPL, avg_steps_success")
    print(f"  - Results directly comparable!")


if __name__ == "__main__":
    main()
