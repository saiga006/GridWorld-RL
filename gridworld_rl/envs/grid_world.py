import pygame
import numpy as np
import random
import sys
import time

# Constants for rendering
CELL_SIZE = 50
MARGIN = 2
WINDOW_PADDING = 20
FPS = 10

# Colors
COLOR_BG = (255, 255, 255)
COLOR_GRID = (200, 200, 200)
COLOR_AGENT = (0, 0, 255)      # Blue
COLOR_OBSTACLE = (255, 0, 0)   # Red
COLOR_GOAL = (0, 255, 0)       # Green
COLOR_DONE_GOAL = (100, 255, 100) # Pale Green
COLOR_TEXT = (0, 0, 0)

class GridWorld:
    def __init__(self, size=10, num_obstacles=5, goal_sequence=None):
        self.size = size
        self.num_obstacles = num_obstacles
        
        # Default goal sequence if none provided
        if goal_sequence is None:
            self.goal_sequence = [(size-1, size-1)]
        else:
            self.goal_sequence = goal_sequence
            
        self.current_goal_idx = 0
        self.window_size = size * (CELL_SIZE + MARGIN) + MARGIN
        self.screen = None
        self.clock = None
        
        self.reset()

    @property
    def action_space_size(self):
        """Returns the number of possible actions. Useful for generic RL algorithms."""
        return 4 # 0: Up, 1: Right, 2: Down, 3: Left

    def reset(self):
        """
        Resets the environment to initial state.
        Returns: initial_state
        """
        self.agent_pos = [0, 0]
        self.current_goal_idx = 0
        self.done = False
        
        # Initialize obstacles at random positions (not on agent or goals)
        self.obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
                if pos != self.agent_pos and tuple(pos) not in self.goal_sequence and pos not in self.obstacles:
                    self.obstacles.append(pos)
                    break
        
        return self._get_state()

    def _get_state(self):
        """
        Returns the current state.
        State includes: (agent_x, agent_y, current_goal_index, obstacle_sensors)
        obstacle_sensors: boolean tuple (up, right, down, left) - 1 if obstacle/wall is there.
        """
        r, c = self.agent_pos
        
        # Sensors: Up, Right, Down, Left
        # Check boundaries or obstacles
        sensors = []
        for move in [0, 1, 2, 3]: # Up, Right, Down, Left
            sr, sc = r, c
            if move == 0: sr -= 1
            elif move == 1: sc += 1
            elif move == 2: sr += 1
            elif move == 3: sc -= 1
            
            # Check collision (Walls or Obstacles)
            is_blocked = 0
            # Wall check
            if sr < 0 or sr >= self.size or sc < 0 or sc >= self.size:
                is_blocked = 1
            # Obstacle check
            elif [sr, sc] in self.obstacles:
                is_blocked = 1
            
            sensors.append(is_blocked)
            
        return (tuple(self.agent_pos), self.current_goal_idx, tuple(sensors))

    def step(self, action):
        """
        Actions:
        0: Up
        1: Right
        2: Down
        3: Left
        """
        if self.done:
            return self._get_state(), 0, True, {}

        # 1. Move Agent
        row, col = self.agent_pos
        if action == 0:   # Up
            row = max(0, row - 1)
        elif action == 1: # Right
            col = min(self.size - 1, col + 1)
        elif action == 2: # Down
            row = min(self.size - 1, row + 1)
        elif action == 3: # Left
            col = max(0, col - 1)
        
        next_pos = [row, col]
        
        # Default Reward
        reward = -1  # Step penalty
        
        # 2. Check Collisions (Walls equivalent to staying in place logic above)
        
        # 3. Check Obstacles (Static or Dynamic checks)
        # Collision with OBSTACLE
        if next_pos in self.obstacles:
            reward = -10 # Penalty for hitting obstacle
            # Optionally stop episode? Or just bounce back?
            # Let's bounce back (stay in current pos)
            next_pos = self.agent_pos 
        
        self.agent_pos = next_pos

        # 4. Check Goals
        current_goal = self.goal_sequence[self.current_goal_idx]
        if tuple(self.agent_pos) == current_goal:
            reward = 100 # Reward for reaching goal
            self.current_goal_idx += 1
            if self.current_goal_idx >= len(self.goal_sequence):
                self.done = True
                reward += 50 # Bonus for finishing sequence
        
        # 5. Move Obstacles (Dynamic)
        # Simple random movement for obstacles
        new_obstacles = []
        for obs in self.obstacles:
            move = random.choice([0, 1, 2, 3, 4]) # 4 is stay
            orow, ocol = obs
            if move == 0: orow = max(0, orow - 1)
            elif move == 1: ocol = min(self.size - 1, ocol + 1)
            elif move == 2: orow = min(self.size - 1, orow + 1)
            elif move == 3: ocol = max(0, ocol - 1)
            
            # Obstacles shouldn't overlap with goals (to prevent impossible states)? 
            # Or other obstacles? For simplicity allow overlap but not with walls.
            if tuple([orow, ocol]) not in self.goal_sequence:
                 new_obstacles.append([orow, ocol])
            else:
                 new_obstacles.append(obs) # Stay if trying to cover a goal
        self.obstacles = new_obstacles
        # Check if obstacle moved INTO agent
        if self.agent_pos in self.obstacles:
            reward -= 10

        result = (self._get_state(), reward, self.done, {})

        return result

    def close(self):
        """Closes the Pygame window if open."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size + 40)) # +40 for text
            pygame.display.set_caption("Grid World RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 18)

        # Event handling for closing window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(COLOR_BG)

        # Draw Grid
        for row in range(self.size):
            for col in range(self.size):
                color = COLOR_GRID
                # Draw grid cell
                pygame.draw.rect(self.screen,
                                 color,
                                 [(MARGIN + CELL_SIZE) * col + MARGIN,
                                  (MARGIN + CELL_SIZE) * row + MARGIN,
                                  CELL_SIZE,
                                  CELL_SIZE])

        # Draw Goals
        for idx, goal in enumerate(self.goal_sequence):
            r, c = goal
            color = COLOR_DONE_GOAL if idx < self.current_goal_idx else COLOR_GOAL
            pygame.draw.rect(self.screen,
                             color,
                             [(MARGIN + CELL_SIZE) * c + MARGIN,
                              (MARGIN + CELL_SIZE) * r + MARGIN,
                              CELL_SIZE,
                              CELL_SIZE])
            # Draw order number
            text = self.font.render(str(idx+1), True, (0,0,0))
            self.screen.blit(text, 
                             ((MARGIN + CELL_SIZE) * c + MARGIN + 15,
                              (MARGIN + CELL_SIZE) * r + MARGIN + 10))

        # Draw Obstacles
        for obs in self.obstacles:
            r, c = obs
            pygame.draw.rect(self.screen,
                             COLOR_OBSTACLE,
                             [(MARGIN + CELL_SIZE) * c + MARGIN,
                              (MARGIN + CELL_SIZE) * r + MARGIN,
                              CELL_SIZE,
                              CELL_SIZE])

        # Draw Agent
        ar, ac = self.agent_pos
        # Draw as a circle
        center_x = (MARGIN + CELL_SIZE) * ac + MARGIN + CELL_SIZE // 2
        center_y = (MARGIN + CELL_SIZE) * ar + MARGIN + CELL_SIZE // 2
        pygame.draw.circle(self.screen, COLOR_AGENT, (center_x, center_y), CELL_SIZE // 2 - 5)

        # Info Text
        status_text = f"Goal: {self.current_goal_idx + 1}/{len(self.goal_sequence)} | Step Penalty: -1"
        img = self.font.render(status_text, True, COLOR_TEXT)
        self.screen.blit(img, (10, self.window_size + 5))

        pygame.display.flip()
        self.clock.tick(FPS)

def main():
    # Example usage / Manual test
    print("Use Arrow Keys to move. Q to quit.")
    
    # Define a sequence of goals
    goals = [(2, 2), (5, 5), (8, 2)]
    env = GridWorld(size=10, num_obstacles=5, goal_sequence=goals)
    env.render()

    running = True
    while running:
        action = None
        # Check for manual keys
        events = pygame.event.get() # Need to process events here for manual control
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action = 0
                elif event.key == pygame.K_RIGHT: action = 1
                elif event.key == pygame.K_DOWN: action = 2
                elif event.key == pygame.K_LEFT: action = 3
                elif event.key == pygame.K_q: running = False
        
        if action is not None:
            state, reward, done, _ = env.step(action)
            print(f"State: {state}, Reward: {reward}, Done: {done}")
            if done:
                print("Sequence Completed! Resetting...")
                time.sleep(1)
                env.reset()
        
        env.render()
    
    pygame.quit()

if __name__ == "__main__":
    main()
