# Import libraries
import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import soundfile as sf


# SECTION Main code here

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(path_data: DictConfig) -> None:
    # print(path_data.test_path.ref_file) # REMOVE_LATER
    # Open reference JSON file
    try:
        with open(path_data.test_path.ref_file, 'r') as ref_file:
            ref_json = json.load(ref_file)
            
            # TODO - Make a automated way to open each spin and target anechoic scene using loop
            print(ref_json[0], "\n")    # REMOVE_LATER
            spin_file_name_to_open = f"{ref_json[0]['signal']}.wav"         # REVIEW - can be named HA_output
            print(spin_file_name_to_open, "\n") # REMOVE_LATER
            target_anechoic_file_name_to_open = f"{ref_json[0]['scene']}_target_anechoic.wav"
            print(target_anechoic_file_name_to_open, "\n")  # REMOVE_LATER



    except FileNotFoundError:
        print(f'File not found: {path_data.test_path.ref_file}')
        return None



# !SECTION

if __name__ == '__main__':
    main()