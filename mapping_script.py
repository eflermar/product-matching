import pandas as pd
import json
import re

print("Loading items_train.csv to build mappings...")
df = pd.read_csv('data/items_train.csv')

def build_mapping(column_name):
    unique_ids = set()
    valid_rows = df[column_name].dropna()
    
    for row in valid_rows:
        # Safely extract all numbers from strings like "[1, 2]" or "1,2"
        numbers = re.findall(r'\d+', str(row))
        for num_str in numbers:
            unique_ids.add(int(num_str))
            
    # Create dictionary mapping: {raw_id_string: continuous_integer}
    # Index 0 is reserved for "Unknown/Missing"
    mapping = {str(raw_id): i + 1 for i, raw_id in enumerate(sorted(unique_ids))}
    return mapping

mappings = {
    'departments': build_mapping('departmentIds'),
    'colors': build_mapping('colorTagIdsString'),
    'brands': build_mapping('brandEditionTagId')
}

with open('categorical_mappings.json', 'w') as f:
    json.dump(mappings, f, indent=4)

print("Mappings saved to categorical_mappings.json!")