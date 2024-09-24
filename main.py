# Import libraries
import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import soundfile as sf

# Functions

@hydra.main(version_base=None, config_path=".", config_name="config")
def load_conf(cfg: DictConfig):
    print(cfg, "\n------\n")
    print(cfg.test_path.root)
    print(Path(cfg.test_path.root))
    return cfg


# SECTION Main code here

def main() -> None:
    # Get path data from config.yaml
    path_data = load_conf()
    print("----\n\n", path_data)
    # print(path_data['test_path']['root'])







# !SECTION

if __name__ == '__main__':
    main()