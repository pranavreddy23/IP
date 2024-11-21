import base64
from groq import Groq
from PIL import Image
import io
import re
from typing import List, Tuple

# ... (keep other existing functions like encode_and_resize_image)

def parse_actions(output: str) -> List[Tuple[str, str]]:
    actions = []
    action_types = ["RESET_MAP", "avoid_areas", "prefer_areas", "PICKUP_GOAL", "PLACE_GOAL"]
    
    # Split the output into sentences
    sentences = re.split(r'(?<=[.!?])\s+', output)
    
    for sentence in sentences:
        for action_type in action_types:
            if action_type.lower() in sentence.lower():
                # Find the index of the action type in the sentence
                start_index = sentence.lower().index(action_type.lower())
                # Extract the description as everything after the action type
                description = sentence[start_index + len(action_type):].strip()
                # Clean up the description
                description = re.sub(r'^[:\s]+', '', description)  # Remove leading colons and spaces
                description = re.sub(r'[.!?,;]+$', '', description)  # Remove trailing punctuation
                actions.append((action_type, description))
                break
    
    return actions


def extract_list_from_string(s: str) -> List[str]:
    return [item.strip().strip('"') for item in s.strip("[]").split(",")]
# Path to your local image file
image_path = "/home/pranav/Pictures/map_f.png"


client = Groq()

def encode_and_resize_image(image_path, max_size=(300, 300)):
    with Image.open(image_path) as img:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.thumbnail(max_size)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

#image_path = "path/to/your/map_image.jpg"
base64_image = encode_and_resize_image(image_path)

prompt = """Analyze this 2D map image and provide a sequence of actions for a robot to navigate from start position to pickup at shelf 3 beside self 2 while avoiding the repair area and not disturbing pedestrians and going in a similar way back to the place at the storage area,empty area and free lane are not restricted to use, suggest actions to update the map for navigation planning. Use only these action types:

1. RESET_MAP
2. avoid_areas = ["area x","area y"]
3. prefer_areas = ["area z", "area w"]: Mark an area in image that is preferred for navigation
4. PICKUP_GOAL: Mark the pickup location
5. PLACE_GOAL: Mark the placement location in the storage area

Provide your response as sequence of actions for pick phase and place phase in a list of python, focusing on updating the map for both the pickup and placement phases of the navigation. Do not include any additional explanations or text outside of this format.o not include specific movement instructions."""

# completion = client.chat.completions.create(
#     model="llama-3.1-70b-versatile",
#     messages=[
#         {
#             "role": "user",
#             "content": f"{prompt}\n\n[Image: data:image/jpeg;base64,{base64_image}]"
#         }
#     ],
#     temperature=0.5,
#     max_tokens=1024,
#     top_p=1,
#     stream=False,
#     stop=None,
# )

# # Capture the output
# output = completion.choices[0].message.content

# # Parse the actions from the captured output
# parsed_actions = parse_actions(output)

# # Print the raw output and parsed actions
# print("Raw output from VLM:")
# print(output)
# print("\nParsed Actions:")
# for action_type, description in parsed_actions:
#     print(f"Action: {action_type}, Description: {description}")

def get_user_input():
    prompt = input("Enter your prompt for the VLM model (press Enter to use the default): ").strip()
    if not prompt:
        prompt = """Analyze this 2D map image and provide a sequence of actions for a robot to navigate from start position to pickup at shelf 3 beside self 2 while avoiding the repair area and not disturbing pedestrians and going in a similar way back to the place at the storage area,empty area and free lane are not restricted to use, suggest actions to update the map for navigation planning. Use only these action types:

1. RESET_MAP
2. avoid_areas = ["area x","area y"]
3. prefer_areas = ["area z", "area w"]: Mark an area in image that is preferred for navigation
4. PICKUP_GOAL: Mark the pickup location
5. PLACE_GOAL: Mark the placement location in the storage area

Provide your response as sequence of actions for pick phase and place phase in a list of python, focusing on updating the map for both the pickup and placement phases of the navigation. Do not include any additional explanations or text outside of this format.o not include specific movement instructions."""
    
    image_path = input("Enter the path to your map image: ").strip()
    if not image_path:
        image_path = "/home/pranav/Pictures/map_f.png"
    
    return prompt, image_path

def get_vlm_actions(image_path, prompt):
    base64_image = encode_and_resize_image(image_path)
    
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n[Image: data:image/jpeg;base64,{base64_image}]"
            }
        ],
        temperature=0.5,
        max_tokens=1024,
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

if __name__ == "__main__":
    user_prompt, user_image_path = get_user_input()
    
    actions = get_vlm_actions(user_image_path, user_prompt)
    
    print("Parsed Actions:")
    if actions:
        for action_type, description in actions:
            print(f"Action: {action_type}, Description: {description}")
    else:
        print("No actions were parsed from the output.")

