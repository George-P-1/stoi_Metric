# Import libraries
import pandas as pd
import numpy as np
import json

def calc_RMSE(stoi_arr, listeners_arr) -> float:
    """
    Calculates Root Mean Squared Error value when given two arrays.
    Order of arguments doesn't matter in this function.
    But remember to normalize before passing the arrays.
    Output is in the same normalized scale as the input arrays.

    Arguments:
        stoi_arr (list[float] or np.ndarray): A list of STOI values
        listeners_arr (list[float] or np.ndarray): A list of true intelligibility values from clarity JSON file

    Returns:
        float: RMSE value
    """
    error = stoi_arr - listeners_arr
    return np.sqrt(np.mean((error) ** 2))

def main() -> None:
    # SECTION - Load Data from csv files 
    # With 'Silent Frame Removal'
    mystoi_csv_path = "mystoi_scores_2025-01-16_09-47-05.csv"
    pystoi_csv_path = "pystoi_scores_2025-01-16_09-47-05.csv"
    true_scores_path = "true_listeners_scores.csv"
    # Without 'Silent Frame Removal'
    # mystoi_csv_path = "mystoi_scores_2025-01-16_10-37-23.csv"
    # pystoi_csv_path = "pystoi_scores_2025-01-16_10-37-23.csv"
    # Only 1000 samples
    # mystoi_csv_path = "mystoi_scores_2025-01-16_17-25-56.csv"
    # pystoi_csv_path = "pystoi_scores_2025-01-16_17-25-56.csv"

    # Import data frame
    mystoi_df = pd.read_csv(mystoi_csv_path)
    pystoi_df = pd.read_csv(pystoi_csv_path)
    true_scores_df = pd.read_csv(true_scores_path)

    # Extract columns
    scene_indices = mystoi_df['Scene_Index'].to_numpy()
    mystoi_scores = mystoi_df['STOI_Value'].to_numpy()
    pystoi_scores = pystoi_df['STOI_Value'].to_numpy()
    true_scores = true_scores_df['true_value'].to_numpy()

    # Check for matching number of rows
    if len(mystoi_df) != len(true_scores):
        if len(mystoi_df) < len(true_scores):
            true_scores = true_scores[:len(mystoi_scores)]            
        elif len(mystoi_df) > len(true_scores):
            raise Exception("Number of rows in mystoi_scores is greater than the number of rows in true_scores.")

    #!SECTION

    # SECTION - Calculate RMSE
    # Normalize so that true scores are between 0 and 1
    true_scores = true_scores/100
    rmse_mystoi = calc_RMSE(mystoi_scores, true_scores)
    rmse_pystoi = calc_RMSE(pystoi_scores, true_scores)

    # Print RMSEs
    print(f'MySTOI RMSE: {rmse_mystoi}')
    print(f'PySTOI RMSE: {rmse_pystoi}')
    print(f'Mystoi RMSE [%]: {rmse_mystoi * 100:.2f}%')
    print(f'PySTOI RMSE [%]: {rmse_pystoi * 100:.2f}%')
    #!SECTION


if __name__ == '__main__':
    main()