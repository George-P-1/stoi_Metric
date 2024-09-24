# Import libraries
import yaml
import json
import soundfile as sf
import os

# Functions

def load_yaml(file_path):
    """
    Load data from YAML file
    """
    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.load(file, Loader=yaml.SafeLoader)
            return yaml_data
    except FileNotFoundError:
        print(f'File not found: {file_path}')
        return None
    except yaml.YAMLError as error:
        print(f'Error parsing YAML: {error}')
        return None


# SECTION Main code here

def main() -> None:
    # Get path data from config.yaml
    path_data = load_yaml("config.yaml")
    print(path_data)
    print(path_data['test_path']['root'])







# !SECTION

if __name__ == '__main__':
    main()