# Import libraries
import numpy as np

# ------------------------------------
# NOTE - How to calculate STOI?
# Resample audios to 10kHz
#           Now audio is in amplitude-time domain
# TF-decompose the signals, i.e. make them in time frequency domain. using moving Hann window for both clean and spin audio. (Hann, 256 samples, 512, 50%, sfft)
#           Now audio is in time-frequency domain
# Remove silent frames - find frame with max energy. Then remove the frames with energy less than 40dB relative to max energy frame. 
# Reconstruct clean and spin audio after the previous step
#           Now not sure if i should reconstruct to ampli-time domain or TF domain. # REVIEW 
# Some shit about 1/3 octave band analysis (1/3 octave, 15 bands, 150Hz, 4.3kHz) # REVIEW
# Compare the clean and spin audio with moving window (Could be related to previous step)
# ...?
# Take the average of the list of values obtained from comparison and return that float value
# ------------------------------------

# SECTION - Constants
# STUB
sr = 10000                  # Sampling frequency
# Preprocessing (TF-Decomposition) - Silent frame removal related
frame_len= 256              # number of samples of Hann-windowed frames
overlap = 0.5*256           # 50% overlap =128 samples both sides of frame. So padding upto 512 samples 
# one-third octave related
octave_bands = 15           # Number of one-third octave bands used
lcf = 150                   # lowest center frequency
hcf = 43000                 # highest center frequency
# Comparison related
frame_time = 386            # 386 or 384 milliseconds. Paper shows both numbers in first few pages.

# SECTION - Functions

# STUB
def compute_stoi(clean_audio, spin_audio, sampling_rate: int) -> float:
    """
    # Short-Time Objective Intelligibility
    Computes STOI by comparing a clean and speech-in-noise audio. 

    Arguments:
        clean_audio: clean speech (target_anechoic)
        spin_audio: Speech-in-Noise
        sampling_rate (int): Assumes both audios have same sampling rate

    Returns:
        stoi_val (float): A value that has monotonic relation with subjective speech intelligibility (actual listeners score). 
        
        Higher STOI value corresponds to better intelligibility.
    """
    # Check if the sampling rate is 10kHz
    # REVIEW - Maybe later the resampling can be done within compute_stoi() function
    if sr != sampling_rate:
        raise Exception("Sampling rate is not {}".format(sr))

    pass


def calc_RMSE(stoi_arr, listeners_arr) -> float:
    """
    Calculates Root Mean Squared Error value when given a list of computed STOI values and list of actual listeners scores. 
    
    Arguments:
        stoi_arr (list[float] or np.ndarray): A list of STOI values
        listeners_arr (list[float] or np.ndarray): A list of listeners scores from clarity JSON file

    Returns:
        float: RMSE value
    """
    error = stoi_arr - listeners_arr
    return np.sqrt(np.mean((error) ** 2))