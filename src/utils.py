import json

def load_json(filepath):
    """
    Loads a JSON file and returns the data as a Python object.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
