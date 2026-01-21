import numpy as np
import random
import pickle
import csv
import os
from grid_world import GridWorld

# Hyperparameters
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998
EPISODES = 3000 
MAX_STEPS = 200 

def get_q_value(q_table, state, action):
    if state not in q_table:
        return 0.0
    return q_table[state][action]

def choose_action(q_table, state, epsilon, env):
    if random.random() < epsilon:
        return random.randint(0, env.action_space_size - 1)
    else:
        if state not in q_table:
            return random.randint(0, env.action_space_size - 1)
        return int(np.argmax(q_table[state]))

def train():
    # Setup Environment
    # Defining a fixed goal sequence for training
    goals = [(2, 2), (5, 5), (8, 2), (1, 8)]
    grid_size = 10
    num_obstacles = 5
    
    env = GridWorld(size=grid_size, num_obstacles=num_obstacles, goal_sequence=goals)
    
    q_table = {}
    epsilon = EPSILON_START
    
    # Setup for Logging
    log_filename = "q_learning_training_log.csv"
    with open(log_filename, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'episode_reward', 'episode_length', 'training_loss', 'exploration_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Training Loop
    print(f"Starting training for {EPISODES} episodes...")
    
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        total_td_error = 0
        td_error_count = 0
        
        while not done and steps < MAX_STEPS:
            action = choose_action(q_table, state, epsilon, env)
            
            next_state, reward, done, _ = env.step(action)
            
            
            if state not in q_table:
                q_table[state] = np.zeros(env.action_space_size)
            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space_size)
            
            # Q-Learning Update
            old_value = q_table[state][action]
            
            # Calculating TD Target and Error
            if done:
                td_target = reward
            else:
                next_max = np.max(q_table[next_state])
                td_target = reward + GAMMA * next_max
                
            td_error = td_target - old_value
            
            # Updating Q-value
            new_value = old_value + ALPHA * td_error
            q_table[state][action] = new_value
            
            # Accumulate TD logs
            total_td_error += abs(td_error)
            td_error_count += 1
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Decay Epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Calculate avg loss for the episode
        avg_loss = total_td_error / td_error_count if td_error_count > 0 else 0
        
        # Log metrics
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['episode', 'episode_reward', 'episode_length', 'training_loss', 'exploration_rate'])
            writer.writerow({
                'episode': episode,
                'episode_reward': episode_reward,
                'episode_length': steps,
                'training_loss': avg_loss,
                'exploration_rate': epsilon
            })
            
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward={episode_reward}, Steps={steps}, Loss={avg_loss:.4f}, Epsilon={epsilon:.4f}")

    # Save Policy
    policy_filename = "q_learning_policy.pkl"
    with open(policy_filename, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"Training complete. Policy saved to {policy_filename}")
    print(f"Training data saved to {log_filename}")

if __name__ == "__main__":
    train()
