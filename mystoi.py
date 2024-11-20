# Import libraries
import numpy as np

# ------------------------------------
# NOTE - How to calculate STOI?
# Resample audios to 10kHz 
# Remove silent frames, using moving Hann window for both clean and spin audio
# Reconstruct clean and spin audio after the previous step
# Some shit about 1/3 octave bands
# Compare the clean and spin audio with moving window (Could be related to previous step)
# ...?
# Take the average of the list of values obtained from comparison and return that float value
# ------------------------------------

# SECTION - Constants
# STUB
window_length = 256
window_padding = 512
overlap = 0.5               # 50% 
octave_bands = 15           # Number of one-third octave bands used
lcf = 150                   # lowest center frequency
hcf = 43000                 # highest center frequency
sr = 10000                  # Sampling frequency


# SECTION - Functions

# STUB
def compute_stoi(clean_audio, spin_audio, sampling_rate) -> float:
    """
    # Short-Time Objective Intelligibility
    Computes STOI by comparing a clean and speech-in-noise audio. 

    Arguments:
        clean_audio: clean speech (target_anechoic)
        spin_audio: Speech-in-Noise
        sampling_rate: Assumes both audios have same sampling rate

    Returns:
        float: A value that has monotonic relation with subjective speech intelligibility (actual listeners score). 
        
        Higher STOI value corresponds to better intelligibility.
    """
    # Check if the sampling rate is 10kHz
    # REVIEW - Maybe later the resampling can be done within compute_stoi() function
    if sr != sampling_rate:
        raise Exception("Sampling rate is not {}".format(sr))


    pass

# STUB
def calc_RMSE(stoi_arr, listeners_arr) -> float:
    """
    Calculates Root Mean Squared Error value when given a list of computed STOI values and list of actual listeners scores. 
    
    Arguments:
        stoi_arr (list[float] or np.ndarray): A list of STOI values
        listeners_arr (list[float] or np.ndarray): A list of listeners scores from clarity JSON file

    Returns:
        float: RMSE value
    """
    pass