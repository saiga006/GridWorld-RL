# GridWorld RL: Q-Learning vs PPO RL Agent performance

A comparative study of classical and deep reinforcement learning algorithms on the task of multi-goal sequential navigation task with dynamic obstacles in a gridworld(Pygame environment).

## Contributors

- [Sai Mukkundan Ramamoorthy](mailto:sai.ramamoorthy@smail.inf.h-brs.de) (for SKRL environment setup and PPO RL Agent)
- [Aaron Cuthinho](mailto:aaron.cuthinho@smail.inf.h-brs.de) (for Grid world environment setup & Q-learning Agent)

---

## Overview

This project was done as part of HBRS Machine Learning Course Project. We implement and then compare two reinforcement learning approaches on a grid world with dynamic obstacles:

- **Q-Learning** — classical tabular RL (model-free, discrete)
- **PPO (Proximal Policy Optimization)** — deep RL with Actor-Critic neural networks

Both agents are trained to navigate a 10×10 grid, visiting **4 goals in sequence** while avoiding **5 dynamically moving obstacles**.

---

## Environment: GridWorld

The environment is a custom 10×10 grid built with Pygame.

### State Space (7-dimensional observation)

| Index | Feature | Description |
| ------- | --------- | ------------- |
| 0 | `agent_x` | Agent row position (0–9) |
| 1 | `agent_y` | Agent column position (0–9) |
| 2 | `goal_idx` | Current goal index (0–3) |
| 3 | `sensor_up` | Obstacle/wall directly above (0 or 1) |
| 4 | `sensor_right` | Obstacle/wall directly right (0 or 1) |
| 5 | `sensor_down` | Obstacle/wall directly below (0 or 1) |
| 6 | `sensor_left` | Obstacle/wall directly left (0 or 1) |

### Action Space (4 discrete actions)

| Action | Direction |
| -------- | ----------- |
| 0 | Up |
| 1 | Right |
| 2 | Down |
| 3 | Left |

### Goal Sequence

```python
GOAL_SEQUENCE = [(2, 2), (5, 5), (8, 2), (1, 8)]  # Same for both algorithms
```
The agent must reach all 4 goals **in order** within 200 steps.

### Reward Structure

| Event | Reward |
| ------- | -------- |
| Reaching a goal | +100 |
| Completing all 4 goals | +50 (sequence bonus) |
| Each timestep | −1 (step penalty) |
| Obstacle collision | −10 (for hitting an obstacle, and if the dynamic obstacle moves onto agent position) |

### Dynamic Obstacles
- 5 obstacles move randomly each step (up/right/down/left/stay)
- They never overlap with goal positions
- They can move into the agent's position, triggering a collision penalty

---

## Results

Both algorithms were evaluated over **100 episodes** on the same goal sequence and environment configuration.

| Metric | Q-Learning | PPO (Deep RL) |
| -------- | ----------- | --------------- |
| **Success Rate** | **100%** | **100%** |
| **SPL (Path Efficiency)** | **0.91** | 0.83 |
| **Avg Steps (Success)** | **32.23** | 35.57 |
| **Collision Rate** | 83% | **65%** |
| **Optimal Episodes (29 steps)** | **31 / 100** | 0 / 100 |
| **Training Time** | ~30 min | **~12 min** |

> Optimal path length = **29 steps** (Manhattan distance through all 4 goals).

- Both algorithms achieved **perfect 100% task completion**.
- **Q-Learning** produced more optimal paths (SPL 0.91 vs 0.83) and reached exact optimal length in 31% of episodes.
- **PPO** navigated more safely with significantly fewer collisions (65% vs 83%).
- **PPO converged by 100K timesteps** — the remaining 300K was spent in stablising the loss curve while maintaining the high cumulative reward.
- Thus we can conclude for grid world with discrete and manageable action-space, Q-learning is better suited than the PPO algorithm to derive an optimal RL policy agent.

---

## Performance Metrics Explained

| Metric | Definition |
| -------- | ----------- |
| **Success Rate** | % of episodes where all 4 goals were reached within the 200-step timeout |
| **Collision Rate** | % of episodes where the agent hit at least one obstacle |
| **SPL** | `(Optimal Steps / Actual Steps) × Success` — path efficiency (1.0 = perfect) |
| **Avg Steps** | Mean timesteps per successful episode (lower = more efficient) |

---

## Project Structure

The repository is divided into two main sections for each algorithm:

```
GridWorld-RL/
├── GridWorld_Q-Learning/          # Q-Learning Implementation
│   ├── evaluate_q_learning.py     # Evaluation & video recording script
│   ├── grid_world.py              # GridWorld environment (Pygame)
│   └── train_q_learning.py        # Q-learning training script
├── gridworld_rl/                  # PPO Implementation
│   ├── configs/
│   │   └── ppo_config.py          # PPO hyperparameter configuration (skrl)
│   ├── envs/
│   │   ├── grid_world.py          # Core GridWorld logic
│   │   └── gym_wrapper.py         # Gym-compatible wrapper for skrl, required for registering the grid world with openai gym environment
│   ├── models/
│   │   └── networks.py            # Actor (PolicyNetwork) & Critic (ValueNetwork)
│   ├── utils/
│   │   └── metrics.py             # Evaluation metrics
│   ├── evaluate.py                # PPO evaluation script
│   ├── train.py                   # PPO training script
│   ├── ppo_eval_log.csv           # PPO evaluation results
│   └── ppo_eval_summary.txt       # PPO evaluation summary
└── README.md
```

---

## Algorithms

### 1. Q-Learning

Classical tabular reinforcement learning. Stores a Q-value for every (state, action) pair.

**Update Rule:**
```
Q(s, a) ← Q(s, a) + α [r + γ · max Q(s', a') − Q(s, a)]
```

**Hyperparameters:**
| Parameter | Value |
| ----------- | ------- |
| Learning Rate (α) | 0.1 |
| Discount Factor (γ) | 0.99 |
| Epsilon Start | 1.0 |
| Epsilon End | 0.05 |
| Epsilon Decay | 0.998 |
| Training Episodes | 3,000 |
| Max Steps per Episode | 200 |

---

### 2. PPO (Proximal Policy Optimization)

Deep RL using a separate Actor-Critic architecture implemented with [skrl](https://skrl.readthedocs.io/).

**PPO Clipped Objective:**
```
L(θ) = E[min(r_t(θ) · Â_t,  clip(r_t(θ), 1−ε, 1+ε) · Â_t)]
```
where `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` and `ε = 0.2`.

**Hyperparameters:**
| Parameter | Value |
| ----------- | ------- |
| Learning Rate | 3.0 × 10⁻⁴ |
| Discount Factor (γ) | 0.99 |
| GAE Lambda (λ) | 0.95 |
| PPO Clip Range (ε) | 0.2 |
| Value Clip | 0.2 |
| Entropy Coefficient | 0.02 |
| Value Loss Scale | 0.5 |
| Max Gradient Norm | 0.5 |
| Rollout Length | 2,048 timesteps/env |
| Learning Epochs | 10 |
| Mini-Batches | 32 |
| Parallel Environments | 16 |
| Effective Batch Size | 32,768 (2,048 × 16) |
| Total Timesteps | 400,000 |
| Optimizer | Adam |

#### Network Architecture

Both Actor and Critic networks share the same layer configuration but are **independently trained**:

```
PolicyNetwork (Actor):   7 → [256, ReLU] → [256, ReLU] → [256, ReLU] → 4
ValueNetwork  (Critic):  7 → [256, ReLU] → [256, ReLU] → [256, ReLU] → 1
```

| Component | Details |
| ----------- | --------- |
| Activation | ReLU |
| Total Parameters | ~35,717 |
| Actor Output | 4 action logits → softmax → probabilities |
| Critic Output | 1 scalar V(s) — expected cumulative reward |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/saiga006/GridWorld-RL.git
cd GridWorld-RL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy pygame opencv-python skrl tensorboard
```

---

## Usage

### Train & Evaluate Q-Learning

```bash
cd GridWorld_Q-Learning

# Train
python train_q_learning.py
# Saves policy to: q_learning_policy.pkl
# Logs to:         q_learning_training_log.csv

# Evaluate
python evaluate_q_learning.py
# Evaluates 100 episodes
# Saves results to: q_learning_eval_log.csv
# Records video to: q_learning_evaluation.mp4
```

### Train & Evaluate PPO

```bash
cd gridworld_rl

# Train
python train.py
# Trains for 400,000 timesteps using 16 parallel envs
# Logs to:          runs/ (TensorBoard)
# Saves model to:   runs/gridworld_ppo_*/final_agent.pt

# Evaluate
python evaluate.py
# Saves results to: ppo_eval_log.csv
```

### Monitor PPO Training

```bash
cd gridworld_rl
tensorboard --logdir runs/
# Open: http://localhost:6006
```

---

## References

- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347
- skrl library: https://skrl.readthedocs.io/
- OpenAI Spinning Up: https://spinningup.openai.com/
