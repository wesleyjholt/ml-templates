import yaml

def read_yaml(filename):
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)