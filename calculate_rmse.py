# Import libraries
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import json

from mystoi import calc_RMSE 

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(path_data: DictConfig) -> None:
    # SECTION - Open result JSON file
    try:
        with open(path_data.test_result_path.result_ref_file, 'r') as res_file:
            res_json = json.load(res_file)

            # Add correctness scores to a list
            true_scores = []

            for res_index in range(len(res_json)):
                true_scores.append(res_json[res_index]['correctness'])

            # Convert to numpy array
            true_scores = np.array(true_scores)

    except FileNotFoundError:
        raise Exception(f'JSON file not found: {path_data.test_result_path.result_ref_file}')
    finally:
        print(f'Finished processing Result JSON file.')
    #!SECTION

    # SECTION - Load Data from csv files 
    mystoi_csv_path = "mystoi_scores_20250116_091620.csv"
    pystoi_csv_path = "pystoi_scores_20250116_091620.csv"

    # Import data frame
    mystoi_df = pd.read_csv(mystoi_csv_path)
    pystoi_df = pd.read_csv(pystoi_csv_path)

    # Extract columns
    scene_indices = mystoi_df['Scene_Index'].to_numpy()
    mystoi_scores = mystoi_df['STOI_Value'].to_numpy()
    pystoi_scores = pystoi_df['STOI_Value'].to_numpy()
    
    # Check for matching number of rows
    if len(mystoi_df) != len(true_scores):
        raise ValueError("The number of rows in the computed STOI file and true scores JSON file must match.")
    
    #!SECTION

    # SECTION - Calculate RMSE
    rmse_mystoi = calc_RMSE(mystoi_scores, true_scores)
    rmse_pystoi = calc_RMSE(pystoi_scores, true_scores)

    # Print RMSEs
    print(f'MySTOI RMSE: {rmse_mystoi}')
    print(f'PySTOI RMSE: {rmse_pystoi}')
    #!SECTION


if __name__ == '__main__':
    main()