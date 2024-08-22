import json
import yaml

def write_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def write_yaml(filename, data):
    with open(filename, 'w') as f:
        yaml.dump(data, f)

def read_yaml(filename):
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)