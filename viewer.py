import random
from pathfinding import a_star, dcip, create_grid, define_zones, assign_costs_based_on_instructions
from pathviz import  generate_random_start_goal
from typing import Tuple, List, Optional, Set
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

# Define zone constants
EMPTY_SPACE = 0
OBSTACLE = 1
WALKWAY = 2
LIVING_ROOM_LEFT = 3
LIVING_ROOM_RIGHT = 4
REPAIR_AREA = 5
ABANDONED_AREA = 6

def plot_algorithm_nodes(
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    expanded_nodes: Set[Tuple[int, int]],
    algorithm_name: str,
    filename: str
):
    """
    Plots the grid with obstacles, start and goal positions,
    and highlights nodes expanded by the specified algorithm.
    """
    # Define color mapping for zones using RGB values
    cmap = colors.ListedColormap([
        'white',      # EMPTY_SPACE (0)
        'black',      # OBSTACLE (1)
        'yellow',     # WALKWAY (2)
        'lightblue',  # LIVING_ROOM_LEFT (3)
        'lightgreen', # LIVING_ROOM_RIGHT (4)
        'orange',     # REPAIR_AREA (5)
        'grey'        # ABANDONED_AREA (6)
    ])
    
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    
    # Plot the grid using imshow with the colormap
    plt.imshow(grid, cmap=cmap, interpolation='nearest')
    
    # Plot start and goal positions
    plt.scatter(start[0], start[1], color='green', s=100, label='Start')
    plt.scatter(goal[0], goal[1], color='red', s=100, label='Goal')
    
    # Plot expanded nodes
    if expanded_nodes:
        nodes_x, nodes_y = zip(*expanded_nodes)
        node_color = 'blue' if algorithm_name == 'A*' else 'purple'
        plt.scatter(nodes_x, nodes_y, marker='.', color=node_color, 
                   s=10, label=f'{algorithm_name} Expanded Nodes', alpha=0.6)
    
    plt.title(f"Nodes Expanded by {algorithm_name}")
    plt.legend(markerscale=4)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig(filename)
    plt.show()
    plt.close()

def main():
    # Grid configuration
    width, height = 150, 90
    obstacle_prob = 0.2
    grid = create_grid(width, height, obstacle_prob)
    grid = define_zones(grid)
    
    # Generate start and goal positions
    start, goal = generate_random_start_goal(grid)
    
    # Define instructions for DCIP
    instructions_output = {
        'prefer_zones': [WALKWAY, LIVING_ROOM_LEFT],
        'avoid_zones': [ABANDONED_AREA, REPAIR_AREA]
    }
    
    # Assign cost grid for DCIP
    cost_grid = assign_costs_based_on_instructions(grid, instructions_output)
    
    # Run A* and plot its expanded nodes
    result_a_star = a_star(grid, start, goal)
    if result_a_star:
        expanded_a_star = result_a_star.get('expanded_positions', set())
        print(f"A* expanded nodes count: {len(expanded_a_star)}")
        plot_algorithm_nodes(
            grid=grid,
            start=start,
            goal=goal,
            expanded_nodes=expanded_a_star,
            algorithm_name='A*',
            filename=f"a_star_nodes_{width}x{height}.png"
        )
    else:
        print("A* failed to find a path.")
    
    # Run DCIP and plot its expanded nodes
    result_dcip = dcip(grid, cost_grid, start, goal, instructions_output)
    if result_dcip:
        expanded_dcip = result_dcip.get('expanded_positions', set())
        print(f"DCIP expanded nodes count: {len(expanded_dcip)}")
        plot_algorithm_nodes(
            grid=grid,
            start=start,
            goal=goal,
            expanded_nodes=expanded_dcip,
            algorithm_name='DCIP',
            filename=f"dcip_nodes_{width}x{height}.png"
        )
    else:
        print("DCIP failed to find a path.")
    
    print("Visualization Completed.")

if __name__ == "__main__":
    main()