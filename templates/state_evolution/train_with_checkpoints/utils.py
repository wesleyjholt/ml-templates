# FEEL FREE TO ADD/REMOVE UTILITIES AS NEEDED

from time import time
import json
import yaml

def compute_elapsed_time(start_time):
    time_in_seconds = time() - start_time
    hours, remainder = divmod(time_in_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:>2.0f} h {minutes:>2.0f} m {seconds:>4.1f} s'

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