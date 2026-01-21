"""
Utility functions for computing evaluation metrics.
ALIGNED with Q-learning evaluation - uses Manhattan distance.
"""
import numpy as np
from typing import List, Tuple


def calculate_manhattan_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """
    Calculate Manhattan distance between two positions.
    MATCHES Q-learning implementation exactly.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def calculate_optimal_path_length_manhattan(start_pos: Tuple[int, int], 
                                           goals: List[Tuple[int, int]]) -> int:
    """
    Calculate optimal path length using Manhattan distance.
    MATCHES Q-learning implementation exactly.
    
    Calculates: Start -> Goal1 -> Goal2 -> ... -> GoalN
    
    Args:
        start_pos: Starting position (row, col)
        goals: List of goal positions in sequence
        
    Returns:
        Total Manhattan distance along the path
    """
    total_dist = 0
    current = start_pos
    for goal in goals:
        total_dist += calculate_manhattan_dist(current, goal)
        current = goal
    return total_dist


def calculate_spl_manhattan(success: bool, 
                           actual_steps: int,
                           optimal_length: int) -> float:
    """
    Calculate SPL using Manhattan distance.
    MATCHES Q-learning: SPL = success * (optimal / max(actual, optimal))
    
    Args:
        success: Whether episode succeeded (reached all goals)
        actual_steps: Actual steps taken
        optimal_length: Optimal path length (Manhattan distance)
        
    Returns:
        SPL value between 0 and 1
    """
    if not success:
        return 0.0
    
    if optimal_length == 0:
        return 1.0 if actual_steps == 0 else 0.0
    
    # Q-learning formula: 1.0 * optimal / max(actual, optimal)
    return 1.0 * optimal_length / max(actual_steps, optimal_length)
