import pandas as pd

# Load the evaluation results
df = pd.read_csv('evaluation_results.csv')

# Convert string representations of lists to actual lists of integers
def convert_to_list(s):
    if isinstance(s, str):
        return [int(num.strip()) for num in s.strip('[]').split(',') if num.strip().isdigit()]
    return []

df['Preferred_Areas'] = df['Preferred_Areas'].apply(convert_to_list)
df['Avoided_Areas'] = df['Avoided_Areas'].apply(convert_to_list)

# Expected Outputs for Each Prompt (Numerical)
EXPECTED_ZONES = {
    "1": {
        "prefer": {2, 0},  # Walkway, Empty Space
        "avoid": {1, 5, 6}  # Obstacle, Repair Area, Abandoned Area
    },
    "2": {
        "prefer": {2},  # Walkway
        "avoid": {1, 5, 6}  # Obstacle, Repair Area, Abandoned Area
    },
    "3": {
        "prefer": {2, 3, 4},  # Walkway, Living Room Left, Living Room Right
        "avoid": {1, 5, 6}  # Obstacle, Repair Area, Abandoned Area
    },
    "4": {
        "prefer": {2, 3, 4},  # Walkway, Living Room Left, Living Room Right
        "avoid": {1, 5, 6}  # Obstacle, Repair Area, Abandoned Area
    }
}


def calculate_instruction_adherence(row):
    prompt_id = str(row['Prompt_ID'])
    expected_prefer = EXPECTED_ZONES[prompt_id]['prefer']
    expected_avoid = EXPECTED_ZONES[prompt_id]['avoid']
    
    actual_prefer = set(row['Preferred_Areas'])
    actual_avoid = set(row['Avoided_Areas'])
    
    prefer_correct = actual_prefer == expected_prefer
    avoid_correct = actual_avoid == expected_avoid
    
    return prefer_correct and avoid_correct

# Apply the function
df['Instruction_Adherence'] = df.apply(calculate_instruction_adherence, axis=1)

# Calculate adherence percentage
instruction_adherence = df['Instruction_Adherence'].mean() * 100
print(f"Instruction Adherence: {instruction_adherence:.2f}%")


def calculate_completeness(row):
    prompt_id = str(row['Prompt_ID'])
    expected_prefer = EXPECTED_ZONES[prompt_id]['prefer']
    expected_avoid = EXPECTED_ZONES[prompt_id]['avoid']
    
    actual_prefer = set(row['Preferred_Areas'])
    actual_avoid = set(row['Avoided_Areas'])
    
    prefer_present = len(actual_prefer) > 0
    avoid_present = len(actual_avoid) > 0
    
    prefer_coverage = expected_prefer.issubset(actual_prefer)
    avoid_coverage = expected_avoid.issubset(actual_avoid)
    
    return (prefer_present and avoid_present) and (prefer_coverage and avoid_coverage)

# Apply the function
df['Completeness_of_Response'] = df.apply(calculate_completeness, axis=1)

# Calculate completeness percentage
completeness = df['Completeness_of_Response'].mean() * 100
print(f"Completeness of Response: {completeness:.2f}%")


def calculate_relevance(row):
    prompt_id = str(row['Prompt_ID'])
    expected_relevance = {
        area: "Relevant" if area in EXPECTED_ZONES[prompt_id]['prefer'] else "Avoid"
        for area in AREA_MAPPING.keys()
    }
    
    actual_prefer = set(row['Preferred_Areas'])
    actual_avoid = set(row['Avoided_Areas'])
    
    # Check preferred areas relevance
    prefer_relevant = all(expected_relevance.get(zone) == "Relevant" for zone in actual_prefer)
    
    # Check avoided areas relevance
    avoid_relevant = all(expected_relevance.get(zone) == "Avoid" for zone in actual_avoid)
    
    return prefer_relevant and avoid_relevant

# Apply the function
df['Relevance_to_Strategy'] = df.apply(calculate_relevance, axis=1)

# Calculate relevance percentage
relevance = df['Relevance_to_Strategy'].mean() * 100
print(f"Relevance to Strategy: {relevance:.2f}%")



print(f"Instruction Adherence: {instruction_adherence:.2f}%")
print(f"Completeness of Response: {completeness:.2f}%")
print(f"Relevance to Strategy: {relevance:.2f}%")


