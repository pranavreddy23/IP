import heapq
import time
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional
import math
import sys
import tracemalloc

class Node:
    def __init__(self, position, parent=None, g=0, h=0, f=0):
        self.position = position  # (x, y)
        self.parent = parent
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost to goal
        self.f = f  # Total cost

    def __lt__(self, other):
        return self.f < other.f

def create_grid(width: int, height: int, obstacle_prob: float=0.2, seed: int=None) -> List[List[int]]:
    if seed is not None:
        random.seed(seed)
    grid = [[0 for _ in range(width)] for _ in range(height)]
    
    # Randomly place obstacles based on obstacle_prob
    for y in range(height):
        for x in range(width):
            if random.random() < obstacle_prob:
                grid[y][x] = 1  # 1 represents an obstacle
    return grid
    

def draw_rectangle(grid: List[List[int]], top_left: Tuple[int, int], bottom_right: Tuple[int, int], zone_type: int):
    for y in range(top_left[1], bottom_right[1] + 1):
        for x in range(top_left[0], bottom_right[0] + 1):
            if 0 <= x < len(grid[0]) and 0 <= y < len(grid):
                grid[y][x] = zone_type

import math
from typing import List, Tuple

# Define zone identifiers
EMPTY_SPACE = 0
OBSTACLE = 1
WALKWAY = 2
LIVING_ROOM_LEFT = 3
LIVING_ROOM_RIGHT = 4
REPAIR_AREA = 5
ABANDONED_AREA = 6


def define_zones(grid: List[List[int]], base_grid_size: Tuple[int, int]=(50,30)) -> List[List[int]]:
    """
    Defines zones within the grid based on its dimensions.
    Zones are scaled proportionally with the grid size.
    
    Zone Definitions:
    - 0: Empty Space
    - 1: Obstacle
    - 2: Walkway
    - 3: Living Room Left
    - 4: Living Room Right
    - 5: Repair Area
    - 6: Abandoned Area
    """
    # Base grid dimensions
    base_width, base_height = base_grid_size
    grid_width = len(grid[0])
    grid_height = len(grid)

    # Calculate scaling factors
    scale_x = grid_width / base_width
    scale_y = grid_height / base_height

    # Define base zones with their coordinates (start_x, start_y, end_x, end_y)
    base_zones = {
        WALKWAY: ((10, 21), (20, 30)),
        LIVING_ROOM_LEFT: ((10, 0), (20, 10)),
        LIVING_ROOM_RIGHT: ((30, 0), (40, 10)),
        REPAIR_AREA: ((10, 10), (20, 20)),
        ABANDONED_AREA: ((30, 10), (40, 20))
    }

    for zone_id, ((start_x, start_y), (end_x, end_y)) in base_zones.items():
        # Scale coordinates
        scaled_start_x = int(start_x * scale_x)
        scaled_start_y = int(start_y * scale_y)
        scaled_end_x = int(end_x * scale_x)
        scaled_end_y = int(end_y * scale_y)

        # Ensure coordinates are within grid boundaries
        scaled_start_x = max(0, min(scaled_start_x, grid_width - 1))
        scaled_start_y = max(0, min(scaled_start_y, grid_height - 1))
        scaled_end_x = max(scaled_start_x + 1, min(scaled_end_x, grid_width))
        scaled_end_y = max(scaled_start_y + 1, min(scaled_end_y, grid_height))

        for y in range(scaled_start_y, scaled_end_y):
            for x in range(scaled_start_x, scaled_end_x):
                grid[y][x] = zone_id

    return grid

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(grid: List[List[int]], position: Tuple[int, int]) -> List[Tuple[int, int]]:
    neighbors = []
    x, y = position
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Up, Down
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
            if grid[ny][nx] != 1:  # Not an obstacle
                neighbors.append((nx, ny))
    return neighbors

def assign_costs_based_on_instructions(grid: List[List[int]], instructions: Dict[str, List[int]]) -> List[List[float]]:
    """
    Assigns traversal costs to the grid based on instructions.
    
    Parameters:
    - grid: The grid with defined zones.
    - instructions: Dictionary with 'prefer_zones' and 'avoid_zones'.
    
    Returns:
    - cost_grid: A grid with traversal costs assigned.
    """
    height = len(grid)
    width = len(grid[0])
    
    # Initialize cost grid with default cost
    cost_grid = [[1.0 for _ in range(width)] for _ in range(height)]
    
    # Assign higher costs to avoid zones
    for zone in instructions.get('avoid_zones', []):
        for y in range(height):
            for x in range(width):
                if grid[y][x] == zone:
                    cost_grid[y][x] = 5.0  # Higher cost for avoidance
    
    # Assign lower costs to prefer zones
    for zone in instructions.get('prefer_zones', []):
        for y in range(height):
            for x in range(width):
                if grid[y][x] == zone:
                    cost_grid[y][x] = 0.5  # Lower cost for preference
    
    return cost_grid

def a_star(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Dict]:
    """
    A* algorithm implementation.
    Returns a dictionary with path and metrics or None if no path found.
    """
    tracemalloc.start()
    start_time = time.time()
    open_list = []
    heapq.heappush(open_list, (0, Node(start)))
    closed_set = set()
    nodes_expanded = 0
    expanded_positions = set() 
    
    while open_list:
        current_f, current_node = heapq.heappop(open_list)
        nodes_expanded += 1
        expanded_positions.add(current_node.position)
        
        if current_node.position == goal:
            path = []
            total_cost = current_node.g
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            path = path[::-1]  # Reverse path
            path_length = len(path) - 1
            tracemalloc.stop()
            end_time = time.time()
            memory_consumption = tracemalloc.get_traced_memory()[1]
            return {
                'path': path,
                'nodes_expanded': nodes_expanded,
                'expanded_positions': expanded_positions,
                'search_time': end_time - start_time,
                'path_cost': total_cost,
                'path_length': path_length,
                'num_turns': count_turns(path),
                'path_smoothness': calculate_smoothness(path),
                'memory_consumption': memory_consumption
            }
        
        closed_set.add(current_node.position)
        
        for neighbor_pos in get_neighbors(grid, current_node.position):
            if neighbor_pos in closed_set:
                continue
            tentative_g = current_node.g + 1  # Assuming uniform cost
            neighbor_node = Node(neighbor_pos, current_node, tentative_g, heuristic(neighbor_pos, goal))
            neighbor_node.f = neighbor_node.g + neighbor_node.h
            
            # Check if neighbor is in open_list with a lower f
            in_open = False
            for open_f, open_node in open_list:
                if neighbor_node.position == open_node.position and neighbor_node.f >= open_node.f:
                    in_open = True
                    break
            if not in_open:
                heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
    
    tracemalloc.stop()
    return None  # No path found

def dcip(grid: List[List[int]], cost_grid: List[List[float]], start: Tuple[int, int], goal: Tuple[int, int], instructions: Dict[str, List[int]]=None) -> Optional[Dict]:
    """
    DCIP algorithm implementation.
    Returns a dictionary with path and metrics or None if no path found.
    """
    tracemalloc.start()
    start_time = time.time()
    open_list = []
    heapq.heappush(open_list, (0, Node(start)))
    closed_set = set()
    nodes_expanded = 0
    replanning_count = 0
    expanded_positions = set()  # Initialize replanning count
    
    while open_list:
        current_f, current_node = heapq.heappop(open_list)
        nodes_expanded += 1
        expanded_positions.add(current_node.position)
        
        if current_node.position == goal:
            path = []
            total_cost = current_node.g
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            path = path[::-1]  # Reverse path
            path_length = len(path) - 1
            tracemalloc.stop()
            end_time = time.time()
            memory_consumption = tracemalloc.get_traced_memory()[1]
            return {
                'path': path,
                'nodes_expanded': nodes_expanded,
                'expanded_positions': expanded_positions, 
                'search_time': end_time - start_time,
                'path_cost': total_cost,
                'path_length': path_length,
                'num_turns': count_turns(path),
                'path_smoothness': calculate_smoothness(path),
                'memory_consumption': memory_consumption,
                'replanning_count': replanning_count
            }
        
        closed_set.add(current_node.position)
        
        for neighbor_pos in get_neighbors(grid, current_node.position):
            if neighbor_pos in closed_set:
                continue
            zone_cost = cost_grid[neighbor_pos[1]][neighbor_pos[0]]
            if zone_cost == float('inf'):
                continue  # Impassable
            tentative_g = current_node.g + zone_cost  # Incorporate zone cost
            neighbor_node = Node(neighbor_pos, current_node, tentative_g, heuristic(neighbor_pos, goal))
            neighbor_node.f = neighbor_node.g + neighbor_node.h
            
            # Check if neighbor is in open_list with a lower f
            in_open = False
            for open_f, open_node in open_list:
                if neighbor_node.position == open_node.position and neighbor_node.f >= open_node.f:
                    in_open = True
                    break
            if not in_open:
                heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
    
    tracemalloc.stop()
    return None  # No path found

def count_turns(path: List[Tuple[int, int]]) -> int:
    """
    Counts the number of direction changes (turns) in the path.
    """
    if len(path) < 3:
        return 0
    turns = 0
    direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for i in range(2, len(path)):
        new_direction = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        if new_direction != direction:
            turns += 1
            direction = new_direction
    return turns

def calculate_smoothness(path: List[Tuple[int, int]]) -> float:
    """
    Calculates the average angle between consecutive path segments.
    Lower values indicate smoother paths.
    """
    if len(path) < 3:
        return 0.0
    total_angle = 0.0
    for i in range(1, len(path)-1):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        x3, y3 = path[i+1]
        v1 = (x2 - x1, y2 - y1)
        v2 = (x3 - x2, y3 - y2)
        dot_prod = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])
        if mag1 * mag2 == 0:
            angle = 0.0
        else:
            angle = math.acos(dot_prod / (mag1 * mag2)) * (180 / math.pi)
        total_angle += angle
    average_angle = total_angle / (len(path) - 2)
    return average_angle

def visualize_paths(grid: List[List[int]], path_a: Optional[List[Tuple[int, int]]], path_dcip: Optional[List[Tuple[int, int]]], start: Tuple[int, int], goal: Tuple[int, int]):
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # Define colors for different zones
    zone_colors = {
        0: 'white',        # Empty Space
        1: 'black',        # Obstacle
        2: 'lightgray',    # Walkway
        3: 'lightblue',    # Living Room Left
        4: 'lightgreen',   # Living Room Right
        5: 'orange',       # Repair Area
        6: 'brown'         # Abandoned Area
    }

    # Draw grid cells based on zone types
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            zone = grid[y][x]
            color = zone_colors.get(zone, 'white')
            rect = patches.Rectangle((x, y), 1, 1, linewidth=0.5, edgecolor='gray', facecolor=color)
            ax.add_patch(rect)
    
    # Function to draw paths
    def draw_path(path: List[Tuple[int, int]], color: str, label: str):
        if path:
            px, py = zip(*path)
            plt.plot([x + 0.5 for x in px], [y + 0.5 for y in py], color=color, linewidth=2, label=label)
    
    # Draw A* Path
    draw_path(path_a, 'blue', 'A* Path')
    
    # Draw DCIP Path
    draw_path(path_dcip, 'red', 'DCIP Path')
    
    # Highlight start and goal
    plt.scatter(start[0] + 0.5, start[1] + 0.5, marker='o', color='green', s=200, label='Start')
    plt.scatter(goal[0] + 0.5, goal[1] + 0.5, marker='X', color='purple', s=200, label='Goal')
    
    # Create legend patches for zones
    legend_patches = [patches.Patch(color=zone_colors[z], label=zone_name) 
                      for z, zone_name in zip(zone_colors.keys(), 
                      ['Empty Space', 'Obstacle', 'Walkway', 
                       'Living Room Left', 'Living Room Right', 
                       'Repair Area', 'Abandoned Area'])]
    
    # Add additional legend entries for paths and start/goal
    legend_patches.extend([
        patches.Patch(color='blue', label='A* Path'),
        patches.Patch(color='red', label='DCIP Path'),
        patches.Patch(color='green', label='Start'),
        patches.Patch(color='purple', label='Goal')
    ])
    
    plt.legend(handles=legend_patches, loc='upper right')
    plt.title('Pathfinding Comparison: A* vs. DCIP')
    plt.xlim(0, len(grid[0]))
    plt.ylim(0, len(grid))
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.grid(True, which='both', color='lightgray', linewidth=0.5)
    plt.show()
    
def get_vlm_actions(image_path: str) -> Dict[str, List[int]]:
    """
    Processes the input image and returns preferred and avoided zones.

    Parameters:
    - image_path: Path to the grid/map image.

    Returns:
    - Dictionary with 'prefer_zones' and 'avoid_zones' lists.
    """
    # TODO: Integrate your multimodal model here
    # Example placeholder output:
    instructions = {
        'prefer_zones': [2,3,4],  # Walkways and Living Rooms
        'avoid_zones': [1,5,6]       # Repair and Abandoned Areas
    }
    return instructions

def main():
    # Grid dimensions
    width, height = 200, 200
    obstacle_prob = 0.1  # Reduced obstacle probability for a clearer path
    
    # Create grid
    grid = create_grid(width, height, obstacle_prob, seed=42)
    
    # Define specific areas
    grid = define_zones(grid)
    
    # Define start and goal positions
    start = (62, 0)   # Ensure this is within a traversable zone
    goal = (131, 156)  # Ensure this is within a traversable zone
    
    # Ensure start and goal are not blocked
    grid[start[1]][start[0]] = 0
    grid[goal[1]][goal[0]] = 0
    
    # Step 1: Generate Instructions from Image
    image_path = "/home/pranav/Pictures/map_f.png"  # Update this path as needed
    instructions_output = get_vlm_actions(image_path)  # Assumes this returns {'prefer_zones': [...], 'avoid_zones': [...]}
    
    print("Instructions from VLM:")
    print(f"Preferred Zones: {instructions_output.get('prefer_zones', [])}")
    print(f"Avoided Zones: {instructions_output.get('avoid_zones', [])}\n")
    
    # Step 2: Assign Traversal Costs Based on Instructions
    cost_grid = assign_costs_based_on_instructions(grid, instructions_output)
    
    # Step 3: Run A* Algorithm
    result_a = a_star(grid, start, goal)
    if result_a:
        print("A* Algorithm Results:")
        print(f"Path: {result_a['path']}")
        print(f"Nodes Expanded: {result_a['nodes_expanded']}")
        print(f"Search Time: {result_a['search_time']:.4f} seconds")
        print(f"Path Cost: {result_a['path_cost']}")
        print(f"Path Length: {result_a['path_length']} steps\n")
    else:
        print("A* Algorithm: No path found.\n")
    
    # Step 4: Run DCIP Algorithm with Instructions
    result_dcip = dcip(grid, cost_grid, start, goal, instructions_output)
    if result_dcip:
        print("DCIP Algorithm Results:")
        print(f"Path: {result_dcip['path']}")
        print(f"Nodes Expanded: {result_dcip['nodes_expanded']}")
        print(f"Search Time: {result_dcip['search_time']:.4f} seconds")
        print(f"Path Cost: {result_dcip['path_cost']}")
        print(f"Path Length: {result_dcip['path_length']} steps\n")
    else:
        print("DCIP Algorithm: No path found.\n")
    
    # Step 5: Visualize the Results
    visualize_paths(grid, result_a['path'] if result_a else None,
                   result_dcip['path'] if result_dcip else None,
                   start, goal)

if __name__ == "__main__":
    main()

