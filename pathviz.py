import random
from pathfinding import a_star, dcip, create_grid, define_zones, assign_costs_based_on_instructions
from typing import Dict, Tuple, List, Optional
import math
import matplotlib.pyplot as plt
import tracemalloc
import time
from pathfinding import (
    a_star, 
    dcip, 
    create_grid, 
    define_zones, 
    assign_costs_based_on_instructions,
    EMPTY_SPACE,
    OBSTACLE,
    WALKWAY,
    LIVING_ROOM_LEFT,
    LIVING_ROOM_RIGHT,
    REPAIR_AREA,
    ABANDONED_AREA
)

# Define a maximum cap for growth factors to prevent plotting issues
MAX_GROWTH_FACTOR = 10.0

class Metrics:
    def __init__(self):
        self.metrics_a_star = {
            'operations': [],       # Nodes Expanded
            'storage': [],          # Memory Consumption (bytes)
        }
        self.metrics_dcip = {
            'operations': [],       # Nodes Expanded
            'storage': [],          # Memory Consumption (bytes)
        }
        self.grid_sizes = []
        self.grid_trials = {}  # To store grid, start, goal, paths for selected trials

def generate_random_start_goal(grid: List[List[int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    height = len(grid)
    width = len(grid[0])
    attempts = 0
    while attempts < 1000:
        start = (random.randint(0, width-1), random.randint(0, height-1))
        goal = (random.randint(0, width-1), random.randint(0, height-1))
        if grid[start[1]][start[0]] == EMPTY_SPACE and grid[goal[1]][goal[0]] == EMPTY_SPACE and start != goal:
            return start, goal
        attempts += 1
    raise ValueError("Failed to generate valid start and goal positions after 1000 attempts.")

def calculate_geometric_mean(values: List[float]) -> float:
    if not values:
        return 1.0
    product = 1.0
    for v in values:
        product *= v
    return math.pow(product, 1/len(values))

def plot_grid(
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    path_a_star: Optional[List[Tuple[int, int]]] = None,
    path_dcip: Optional[List[Tuple[int, int]]] = None,
    title: str = "Grid Visualization",
    filename: str = "grid_visualization.png"
):
    """
    Plots the grid with different zones, start and goal positions, and optional paths for A* and DCIP.

    Parameters:
    - grid: 2D list representing the grid with zone identifiers.
    - start: Tuple indicating the start position (x, y).
    - goal: Tuple indicating the goal position (x, y).
    - path_a_star: List of tuples representing the path found by A* (optional).
    - path_dcip: List of tuples representing the path found by DCIP (optional).
    - title: Title of the plot.
    - filename: Name of the file to save the plot.
    """
    import numpy as np
    from matplotlib import colors  # Correct import for colors

    # Define color mapping for zones
    color_mapping = {
        EMPTY_SPACE: 'white',
        OBSTACLE: 'black',
        WALKWAY: 'yellow',
        LIVING_ROOM_LEFT: 'lightblue',
        LIVING_ROOM_RIGHT: 'lightgreen',
        REPAIR_AREA: 'orange',
        ABANDONED_AREA: 'grey'
    }

    # Create a color grid
    height = len(grid)
    width = len(grid[0])
    color_grid = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            zone = grid[y][x]
            color = color_mapping.get(zone, 'white')  # Default to white if zone not found
            color_rgb = colors.to_rgb(color)  # Corrected usage
            color_grid[y, x] = color_rgb

    plt.figure(figsize=(12, 8))
    plt.imshow(color_grid, origin='lower')

    # Mark start and goal positions
    plt.scatter(start[0], start[1], marker='o', color='green', s=100, label='Start')
    plt.scatter(goal[0], goal[1], marker='X', color='red', s=100, label='Goal')

    # Plot paths if provided
    if path_a_star:
        path_a_star_x, path_a_star_y = zip(*path_a_star)
        plt.plot(path_a_star_x, path_a_star_y, color='blue', linewidth=2, label='A* Path')

    if path_dcip:
        path_dcip_x, path_dcip_y = zip(*path_dcip)
        plt.plot(path_dcip_x, path_dcip_y, color='purple', linewidth=2, label='DCIP Path')

    plt.title(title)
    plt.legend()
    plt.grid(False)
    plt.axis('off')  # Hide axis
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

def run_experiments(grid_sizes: List[Tuple[int, int]], trials: int = 5, obstacle_prob: float = 0.2) -> Metrics:
    metrics = Metrics()

    # Define manual instructions (prefer and avoid zones)
    instructions_output = {
        'prefer_zones': [WALKWAY, LIVING_ROOM_LEFT, LIVING_ROOM_RIGHT],
        'avoid_zones': [REPAIR_AREA, ABANDONED_AREA]
    }

    for size in grid_sizes:
        width, height = size
        metrics.grid_sizes.append(size)
        total_operations_a_star = 0
        total_storage_a_star = 0
        total_operations_dcip = 0
        total_storage_dcip = 0

        # Determine the middle trial index
        middle_trial = trials // 2  # Zero-based index

        for trial in range(trials):
            # Create and define zones
            grid = create_grid(width, height, obstacle_prob, seed=trial)
            grid = define_zones(grid)
            try:
                start, goal = generate_random_start_goal(grid)
            except ValueError as ve:
                print(f"Size {width}x{height}, Trial {trial+1}: {ve}")
                # Assign 'MAX_GROWTH_FACTOR' to denote failure in this trial
                total_operations_a_star += MAX_GROWTH_FACTOR
                total_storage_a_star += MAX_GROWTH_FACTOR
                total_operations_dcip += MAX_GROWTH_FACTOR
                total_storage_dcip += MAX_GROWTH_FACTOR
                continue

            # Assign cost grid for DCIP
            cost_grid = assign_costs_based_on_instructions(grid, instructions_output)

            # Run A* and measure memory
            tracemalloc.start()
            start_time = time.time()
            result_a_star = a_star(grid, start, goal)
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            if result_a_star:
                operations_a = result_a_star['nodes_expanded']
                path_a_star = result_a_star.get('path')  # Assuming 'path' key exists
                total_operations_a_star += operations_a
                total_storage_a_star += peak  # Using peak memory during A*
            else:
                total_operations_a_star += MAX_GROWTH_FACTOR
                total_storage_a_star += MAX_GROWTH_FACTOR
                print(f"A* failed to find a path for size {width}x{height}, trial {trial+1}")
                path_a_star = None

            # Run DCIP and measure memory
            tracemalloc.start()
            start_time = time.time()
            result_dcip = dcip(grid, cost_grid, start, goal, instructions_output)
            end_time = time.time()
            current_d, peak_d = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            if result_dcip:
                operations_d = result_dcip['nodes_expanded']
                path_dcip = result_dcip.get('path')  # Assuming 'path' key exists
                total_operations_dcip += operations_d
                total_storage_dcip += peak_d  # Using peak memory during DCIP
            else:
                total_operations_dcip += MAX_GROWTH_FACTOR
                total_storage_dcip += MAX_GROWTH_FACTOR
                print(f"DCIP failed to find a path for size {width}x{height}, trial {trial+1}")
                path_dcip = None

            # If it's the middle trial, store data for plotting
            if trial == middle_trial:
                metrics.grid_trials[size] = {
                    'grid': grid,
                    'start': start,
                    'goal': goal,
                    'path_a_star': path_a_star,
                    'path_dcip': path_dcip
                }

        # Calculate average operations and storage, handling 'inf' appropriately
        avg_operations_a_star = total_operations_a_star / trials
        avg_storage_a_star = total_storage_a_star / trials
        avg_operations_dcip = total_operations_dcip / trials
        avg_storage_dcip = total_storage_dcip / trials

        metrics.metrics_a_star['operations'].append(avg_operations_a_star)
        metrics.metrics_a_star['storage'].append(avg_storage_a_star)
        metrics.metrics_dcip['operations'].append(avg_operations_dcip)
        metrics.metrics_dcip['storage'].append(avg_storage_dcip)

        print(f"Completed Size {width}x{height}")

    return metrics

def plot_nodes_and_storage(metrics: Metrics, grid_sizes: List[Tuple[int, int]]):
    """
    Plots Nodes Expanded and Memory Consumption for A* and DCIP algorithms across different grid sizes.

    Parameters:
    - metrics: Metrics object containing the collected data.
    - grid_sizes: List of grid sizes used in experiments.
    """
    scaling_factors = [size[0] for size in grid_sizes]  # Assuming square grids for simplicity

    # Plot Nodes Expanded
    plt.figure(figsize=(10, 6))
    plt.plot(scaling_factors, metrics.metrics_a_star['operations'], marker='o', label='A* Nodes Expanded')
    plt.plot(scaling_factors, metrics.metrics_dcip['operations'], marker='s', label='DCIP Nodes Expanded')
    plt.title('Nodes Expanded vs Grid Size')
    plt.xlabel('Grid Width')
    plt.ylabel('Nodes Expanded')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('nodes_expanded_comparison.png')
    plt.show()
    plt.close()

    # Plot Memory Consumption
    plt.figure(figsize=(10, 6))
    plt.plot(scaling_factors, metrics.metrics_a_star['storage'], marker='o', label='A* Memory Consumption')
    plt.plot(scaling_factors, metrics.metrics_dcip['storage'], marker='s', label='DCIP Memory Consumption')
    plt.title('Memory Consumption vs Grid Size')
    plt.xlabel('Grid Width')
    plt.ylabel('Memory Consumption (bytes)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('memory_consumption_comparison.png')
    plt.show()
    plt.close()

def main():
    # Define grid sizes (width x height) based on scaling factors 1,2,4,6,8,10
    base_width, base_height = 50, 30
    scaling_factors = [1, 2, 4, 6, 8, 10]
    grid_sizes = [(base_width * factor, base_height * factor) for factor in scaling_factors]

    trials = 5             # Reduced number of trials
    obstacle_prob = 0.2    # Obstacle probability

    print("Starting experiments across multiple grid sizes...")
    metrics = run_experiments(grid_sizes, trials, obstacle_prob)

    print("\nGenerating Plots for Nodes Expanded and Memory Consumption...")
    plot_nodes_and_storage(metrics, grid_sizes)

    # Plot the stored grid trials
    for size, trial_data in metrics.grid_trials.items():
        plot_grid(
            grid=trial_data['grid'],
            start=trial_data['start'],
            goal=trial_data['goal'],
            path_a_star=trial_data['path_a_star'],
            path_dcip=trial_data['path_dcip'],
            title=f"Grid {size[0]}x{size[1]}, Middle Trial",
            filename=f"grid_{size[0]}x{size[1]}_middle_trial.png"
        )

    print("\nEvaluation Completed.")

if __name__ == "__main__":
    main()