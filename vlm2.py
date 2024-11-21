import base64
from groq import Groq
from PIL import Image
import io
import re
from typing import List, Tuple, Dict
import csv
import argparse
import time  # Import the time module for delays

def encode_and_resize_image(image_path, max_size=(300, 300)):
    with Image.open(image_path) as img:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.thumbnail(max_size)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def parse_actions(output: str) -> Dict[str, List[str]]:
    """
    Parses the VLM model output to extract preferred and avoided areas.

    Parameters:
    - output: The raw string output from the VLM model.

    Returns:
    - A dictionary containing lists of preferred areas and avoided areas.
      Example:
      {
          'prefer_areas': ['Walkway', 'Living Room Left'],
          'avoid_areas': ['Repair Area', 'Abandoned Area']
      }
    """
    valid_zones = {
        'Empty Space', 'Obstacle', 'Walkway', 
        'Living Room Left', 'Living Room Right', 
        'Repair Area', 'Abandoned Area'
    }
    
    actions = {
        'prefer_areas': [],
        'avoid_areas': []
    }
    # Use regex to find Preferred Areas
    prefer_pattern = r'Preferred Areas\s*=\s*\[(.*?)\]'
    prefer_match = re.search(prefer_pattern, output, re.IGNORECASE | re.DOTALL)
    if prefer_match:
        prefer_items = re.findall(r'"(.*?)"|\'(.*?)\'', prefer_match.group(1))
        actions['prefer_areas'] = [
            item for sublist in prefer_items for item in sublist 
            if item and item in valid_zones
        ]
    
    # Use regex to find Avoided Areas
    avoid_pattern = r'Avoided Areas\s*=\s*\[(.*?)\]'
    avoid_match = re.search(avoid_pattern, output, re.IGNORECASE | re.DOTALL)
    if avoid_match:
        avoid_items = re.findall(r'"(.*?)"|\'(.*?)\'', avoid_match.group(1))
        actions['avoid_areas'] = [
            item for sublist in avoid_items for item in sublist 
            if item and item in valid_zones
        ]
    
    return actions

def get_user_input() -> Tuple[str, str]:
    prompts = {
        "1": """Analyze this 2D grid map with the following zone labels:

0: Empty Space
1: Obstacle
2: Walkway
3: Living Room Left
4: Living Room Right
5: Repair Area
6: Abandoned Area

Strategy: Navigate as quickly as possible from the start to the goal.

Provide only the preferred and avoided areas for the robot to navigate. The response must be in valid Python list format without any additional text.

Preferred Areas = [...]
Avoided Areas = [...]""",
        
        "2": """Analyze this 2D grid map with the following zone labels:

0: Empty Space
1: Obstacle
2: Walkway
3: Living Room Left
4: Living Room Right
5: Repair Area
6: Abandoned Area

Strategy: Minimize the number of turns required to navigate from the start to the goal.

Provide only the preferred and avoided areas for the robot to navigate. The response must be in valid Python list format without any additional text.

Preferred Areas = [...]
Avoided Areas = [...]""",
        
        "3": """Analyze this 2D grid map with the following zone labels:

0: Empty Space
1: Obstacle
2: Walkway
3: Living Room Left
4: Living Room Right
5: Repair Area
6: Abandoned Area

Strategy: Maximize safety by avoiding hazardous areas during navigation from the start to the goal.

Provide only the preferred and avoided areas for the robot to navigate. The response must be in valid Python list format without any additional text.

Preferred Areas = [...]
Avoided Areas = [...]""",
        
        "4": """Analyze this 2D grid map with the following zone labels:

0: Empty Space
1: Obstacle
2: Walkway
3: Living Room Left
4: Living Room Right
5: Repair Area
6: Abandoned Area

Strategy: Balance efficiency and safety by prioritizing walkways and living areas while avoiding hazardous zones.

Provide only the preferred and avoided areas for the robot to navigate. The response must be in valid Python list format without any additional text.

Preferred Areas = [...]
Avoided Areas = [...]"""
    }

    print("Select a navigation strategy:")
    print("1. Prioritize Speed")
    print("2. Minimize Turns")
    print("3. Maximize Safety")
    print("4. Balance Efficiency and Safety")

    choice = input("Enter choice (1-4): ").strip()
    selected_prompt = prompts.get(choice, prompts["1"])
    
    if choice not in prompts:
        print("Invalid choice. Using the default prompt.\n")
    
    image_path = input("Enter the path to your grid map image: ").strip()
    if not image_path:
        image_path = "/home/pranav/Pictures/dcip.png"
    
    return selected_prompt, image_path

def evaluate_model(prompts: Dict[str, str], image_path: str, iterations: int, output_file: str):
    """
    Evaluates the VLM by looping through each prompt multiple times and storing the outputs.
    
    Parameters:
    - prompts: Dictionary of prompts with their respective IDs.
    - image_path: Path to the grid map image.
    - iterations: Number of times to execute each prompt.
    - output_file: Path to the CSV file to store results.
    """
    base64_image = encode_and_resize_image(image_path)
    client = Groq()  # Initialize your Groq client appropriately

    # Prepare CSV file
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Prompt_ID', 'Strategy', 'Iteration', 'Raw_Output', 'Preferred_Areas', 'Avoided_Areas']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for prompt_id, prompt_text in prompts.items():
            strategy_line = prompt_text.splitlines()[5].strip()  # Assuming Strategy is on line 6 (0-indexed)
            print(f"Evaluating Prompt {prompt_id}: {strategy_line}")
            for i in range(1, iterations + 1):
                print(f"  Iteration {i}/{iterations}")
                # Create the message payload
                message_content = f"{prompt_text}\n\n[Image: data:image/jpeg;base64,{base64_image}]"
                
                # Send request to the VLM
                try:
                    completion = client.chat.completions.create(
                        model="mixtral-8x7b-32768",
                        messages=[
                            {
                                "role": "user",
                                "content": message_content
                            }
                        ],
                        temperature=0.5,
                        max_tokens=512,
                        top_p=1,
                        stream=False,
                        stop=None,
                    )
                except Exception as e:
                    print(f"    Error during VLM request: {e}")
                    continue  # Skip to the next iteration
                
                output = completion.choices[0].message.content
                print(f"    Raw Output: {output}")
                
                # Parse the actions
                parsed_actions = parse_actions(output)
                
                # Write to CSV
                writer.writerow({
                    'Prompt_ID': prompt_id,
                    'Strategy': strategy_line,
                    'Iteration': i,
                    'Raw_Output': output.replace('\n', ' '),  # Replace newlines for CSV
                    'Preferred_Areas': parsed_actions.get('prefer_areas', []),
                    'Avoided_Areas': parsed_actions.get('avoid_areas', [])
                })
                
                # Delay to respect rate limits (2-3 seconds between prompts)
                time.sleep(4)  # 2.5 seconds delay
    print(f"\nEvaluation completed. Results are stored in {output_file}.")

def get_vlm_actions(image_path: str, prompt: str) -> Dict[str, List[str]]:
    base64_image = encode_and_resize_image(image_path)

    client = Groq()  # Initialize your Groq client appropriately

    completion = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n[Image: data:image/jpeg;base64,{base64_image}]"
            }
        ],
        temperature=0.5,
        max_tokens=512,
        top_p=1,
        stream=False,
        stop=None,
    )

    output = completion.choices[0].message.content
    print("Raw VLM output:")
    print(output)
    print("\n")

    parsed_actions = parse_actions(output)

    return parsed_actions

def main():
    parser = argparse.ArgumentParser(description="VLM Evaluation for Robot Navigation")
    parser.add_argument('--evaluate', action='store_true', help='Run the evaluation mode.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations per prompt in evaluation mode.')
    parser.add_argument('--output', type=str, default='evaluation_results.csv', help='Output CSV file for evaluation results.')
    args = parser.parse_args()
    
    prompts = {
        "1": """Analyze this 2D grid map with the following zone labels:

0: Empty Space
1: Obstacle
2: Walkway
3: Living Room Left
4: Living Room Right
5: Repair Area
6: Abandoned Area

Strategy: Navigate as quickly as possible from the start to the goal.

Provide only the preferred and avoided areas for the robot to navigate. The response must be in valid Python list format without any additional text.

Preferred Areas = [...]
Avoided Areas = [...]""",
        
        "2": """Analyze this 2D grid map with the following zone labels:

0: Empty Space
1: Obstacle
2: Walkway
3: Living Room Left
4: Living Room Right
5: Repair Area
6: Abandoned Area

Strategy: Minimize the number of turns required to navigate from the start to the goal.

Provide only the preferred and avoided areas for the robot to navigate. The response must be in valid Python list format without any additional text.

Preferred Areas = [...]
Avoided Areas = [...]""",
        
        "3": """Analyze this 2D grid map with the following zone labels:

0: Empty Space
1: Obstacle
2: Walkway
3: Living Room Left
4: Living Room Right
5: Repair Area
6: Abandoned Area

Strategy: Maximize safety by avoiding hazardous areas during navigation from the start to the goal.

Provide only the preferred and avoided areas for the robot to navigate. The response must be in valid Python list format without any additional text.

Preferred Areas = [...]
Avoided Areas = [...]""",
        
        "4": """Analyze this 2D grid map with the following zone labels:

0: Empty Space
1: Obstacle
2: Walkway
3: Living Room Left
4: Living Room Right
5: Repair Area
6: Abandoned Area

Strategy: Balance efficiency and safety by prioritizing walkways and living areas while avoiding hazardous zones.

Provide only the preferred and avoided areas for the robot to navigate. The response must be in valid Python list format without any additional text.

Preferred Areas = [...]
Avoided Areas = [...]"""
    }

    if args.evaluate:
        image_path = input("Enter the path to your grid map image: ").strip()
        if not image_path:
            image_path = "/home/pranav/Pictures/dcip.png"
        evaluate_model(prompts, image_path, args.iterations, args.output)
    else:
        # Existing interactive mode
        selected_prompt, image_path = get_user_input()
        actions = get_vlm_actions(image_path, selected_prompt)
        
        print("Parsed Actions:")
        if actions:
            if actions.get('prefer_areas'):
                print(f"Preferred Areas: {actions['prefer_areas']}")
            if actions.get('avoid_areas'):
                print(f"Avoided Areas: {actions['avoid_areas']}")
        else:
            print("No actions were parsed from the output.")

if __name__ == "__main__":
    main()