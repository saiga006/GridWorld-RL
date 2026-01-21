import numpy as np
import pickle
import csv
import os
import time
import cv2
import pygame
from grid_world import GridWorld

# Eval Config
EVAL_EPISODES = 100
MAX_STEPS = 200 
RENDER = False 

def load_policy(filename="q_learning_policy.pkl"):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run training first.")
        return None
    with open(filename, 'rb') as f:
        return pickle.load(f)

def choose_action_greedy(q_table, state, env):
    if state not in q_table:
        # Fallback for unseen states: random or stay
        return np.random.randint(0, env.action_space_size) 
    return int(np.argmax(q_table[state]))

def calculate_manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def calculate_optimal_path_length(start_pos, goals):
    """
    Calculates the lengths of the path: 
    Start -> Goal1 -> Goal2 ... -> GoalN
    using Manhattan distance.
    """
    total_dist = 0
    current = start_pos
    for g in goals:
        total_dist += calculate_manhattan_dist(current, g)
        current = g
    return total_dist

def evaluate():
    q_table = load_policy()
    if q_table is None:
        return
        
    # Same config as training
    goals = [(2, 2), (5, 5), (8, 2), (1, 8)]
    grid_size = 10
    num_obstacles = 5
    
    env = GridWorld(size=grid_size, num_obstacles=num_obstacles, goal_sequence=goals)
    
    # Video Recording Setup
    video_filename = "q_learning_evaluation.mp4"
    # Dimensions from GridWorld logic
    width = env.window_size
    height = env.window_size + 40
    fps = 10
    # Initializing VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    print(f"Recording video to {video_filename}...")
    
    # Metrics
    total_successes = 0
    total_collisions = 0
    spl_accum = 0
    steps_success_accum = 0
    
    # Calculate optimal length once (static sequence)
    # Start at 0,0
    optimal_length = calculate_optimal_path_length((0,0), goals)
    print(f"Optimal Path Length for sequence: {optimal_length}")
    
    log_filename = "q_learning_eval_log.csv"
    with open(log_filename, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'success', 'collision', 'steps', 'spl']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        print(f"Starting evaluation for {EVAL_EPISODES} episodes...")
        
        for i in range(1, EVAL_EPISODES + 1):
            state = env.reset()
            done = False
            steps = 0
            collision_occurred = False
            
            while not done and steps < MAX_STEPS:
                # Render and Capture
                env.render()
                if env.screen is not None:
                    # Capture pixels: (Width, Height, 3)
                    pixels = pygame.surfarray.array3d(env.screen)
                    # Transpose to (Height, Width, 3) for OpenCV
                    frame = np.transpose(pixels, (1, 0, 2))
                    # Convert RGB (Pygame) to BGR (OpenCV)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame)
                
                action = choose_action_greedy(q_table, state, env)
                next_state, reward, done, info = env.step(action)
                
                if reward <= -10: # -10 is obstacle penalty
                     collision_occurred = True
                
                state = next_state
                steps += 1
            
            is_success = done and steps < MAX_STEPS
            
            
            if is_success:
                total_successes += 1
                steps_success_accum += steps
                # SPL: Success (1) * Optimal / Max(Actual, Optimal)
                spl = 1.0 * optimal_length / max(steps, optimal_length)
            else:
                spl = 0.0
            
            if collision_occurred:
                total_collisions += 1
                
            spl_accum += spl
            
            # Log episode
            writer.writerow({
                'episode': i,
                'success': 1 if is_success else 0,
                'collision': 1 if collision_occurred else 0,
                'steps': steps,
                'spl': spl
            })
            
    # Final Eval Metrics
    success_rate = total_successes / EVAL_EPISODES
    collision_rate = total_collisions / EVAL_EPISODES
    avg_spl = spl_accum / EVAL_EPISODES
    avg_steps_success = (steps_success_accum / total_successes) if total_successes > 0 else 0
    
    print("-" * 30)
    print(f"Evaluation Results ({EVAL_EPISODES} Episodes)")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Collision Rate: {collision_rate:.2f}")
    print(f"SPL: {avg_spl:.2f}")
    print(f"Avg Steps (Success): {avg_steps_success:.2f}")
    print("-" * 30)
    
    video_writer.release()
    print(f"Video saved to {video_filename}")
    print(f"Evaluation data saved to {log_filename}")

    
    with open("q_learning_eval_summary.txt", "w") as f:
        f.write(f"success_rate,{success_rate}\n")
        f.write(f"collision_rate,{collision_rate}\n")
        f.write(f"SPL,{avg_spl}\n")
        f.write(f"avg_steps_success,{avg_steps_success}\n")

if __name__ == "__main__":
    evaluate()
