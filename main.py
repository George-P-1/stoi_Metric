# Import libraries
import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import soundfile as sf


# SECTION Main code here

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(path_data: DictConfig) -> None:
    print(path_data, "\n------\n")
    print(path_data.test_path.root)
    print(Path(path_data.test_path.root))



# !SECTION

if __name__ == '__main__':
    main()