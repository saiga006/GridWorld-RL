"""
Gymnasium wrapper for GridWorld environment.
Makes the environment compatible with skrl library.
Tracks collisions for evaluation metrics.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .grid_world import GridWorld


class GridWorldGym(gym.Env):
    """Gym-compatible wrapper for GridWorld environment."""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, size=10, num_obstacles=5, goal_sequence=None, 
                 max_steps=200, render_mode=None):
        """
        Args:
            size: Grid size (size x size)
            num_obstacles: Number of dynamic obstacles
            goal_sequence: List of (row, col) tuples for goals
            max_steps: Maximum steps per episode
            render_mode: 'human' or 'rgb_array'
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.collision_count = 0
        
        if goal_sequence is None:
            goal_sequence = [(size-1, size-1)]
        
        self.env = GridWorld(size, num_obstacles, goal_sequence)
        
        # Define observation space
        # [agent_x, agent_y, goal_idx, sensor_up, sensor_right, sensor_down, sensor_left]
        self.observation_space = spaces.Box(
            low=0,
            high=max(size-1, len(goal_sequence)),
            shape=(7,),
            dtype=np.float32
        )
        
        # Define action space (4 discrete actions)
        self.action_space = spaces.Discrete(4)
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.collision_count = 0
        state = self.env.reset()
        obs = self._state_to_obs(state)
        
        return obs, {}
    
    def step(self, action):
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Take action in environment
        state, reward, done, info = self.env.step(int(action))
        obs = self._state_to_obs(state)
        
        # Track collisions - same logic as Q-learning (reward == -10)
        if reward <= -10:
            self.collision_count += 1
        
        # Check truncation (max steps reached)
        truncated = self.current_step >= self.max_steps
        
        # Additional info for evaluation
        info['current_goal_idx'] = self.env.current_goal_idx
        info['goals_reached'] = self.env.current_goal_idx
        info['total_goals'] = len(self.env.goal_sequence)
        info['collision_count'] = self.collision_count
        info['agent_pos'] = tuple(self.env.agent_pos)
        
        return obs, float(reward), done, truncated, info
    
    def _state_to_obs(self, state):
        """Convert internal state to observation vector."""
        pos, goal_idx, sensors = state
        return np.array(
            [pos[0], pos[1], goal_idx] + list(sensors),
            dtype=np.float32
        )
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            self.env.render()
        return None
    
    def close(self):
        """Close the environment."""
        self.env.close()


def make_gridworld_env(size=10, num_obstacles=5, goal_sequence=None, max_steps=200):
    """Factory function to create GridWorld environment instances."""
    return GridWorldGym(
        size=size,
        num_obstacles=num_obstacles,
        goal_sequence=goal_sequence,
        max_steps=max_steps
    )
