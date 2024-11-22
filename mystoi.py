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
# 1/3 octave band analysis (1/3 octave, 15 bands, 150Hz, 4.3kHz) # REVIEW
# Compare the clean and spin audio with moving window (Could be related to previous step)
# ...?
# Take the average of the list of values obtained from comparison and return that float value
# ------------------------------------

# SECTION - Constants
# STUB
SR = 10000                  # Sampling rate/frequency
# Preprocessing (TF-Decomposition) - Silent frame removal and frames related
TRUE_FRAME_LEN = 256        # number of samples of Hann-windowed frames
OVERLAP = 0.5*256           # 50% overlap =128 samples both sides of frame. So padding upto 512 samples 
FRAME_LEN = 512             # =256 + OVERLAP * 2
# one-third octave related
OCTAVE_BANDS = 15           # Number of one-third octave bands used
LCF = 150                   # lowest center frequency. Estimates higher c.f. is approx. 4.3kHz
# Comparison related
FRAME_TIME = 386            # 386 or 384 milliseconds. Paper shows both numbers in first few pages.

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


def one_third_octaves(sr: int, frame_len: int, num_bands: int, lcf):
    """
    Creates center frequencies of bands of one-third octaves and a one-third octave band matrix. 

    Arguments:
        sr (int): Sampling rate
        frame_len (int): Number of samples in a frame of signal
        num_bands (int): Number of one-third octave bands
        lcf : Lowest center frequency

    Returns:
        (obm, cf) (np.ndarray, np.ndarray): Octave Band Matrix and Center frequencies
    """
    octave_kind = 3 # third octave
    
    # Create the frequency axis based on number of samples in a frame and sampling rate
    dft_bins = np.linspace(0, sr, frame_len + 1)
    # the +1 is because sr/2 should be a frequency in the array.
    dft_bins = dft_bins[0 : int(frame_len/2) + 1]    # Take only Nyquist frequencies
    # when taking fourier transform, the half result is regarding negative frequencies that
    # doesn't matter in the real signal. So the dft_bins only need Nyquist frequencies
    
    bands = np.array(range(num_bands))  # list for 15 bands
    cfs = np.power(2.0, (bands / octave_kind)) * lcf # Center frequencies of bands
    # REVIEW - The paper mentions highest center frequency is approx. 4.3kHz but it isnt when calculated.
    ube = lcf * np.power(2.0, (2 * bands + 1) / (2*octave_kind))  # upper band edge
    # NOTE - However, the upper band edge of highest band is approx. 4.3kHz
    lbe = lcf * np.power(2.0, (2 * bands - 1) / (2*octave_kind))  # lower band edge
    
    obm = np.zeros((num_bands, len(dft_bins)))  # Octave Band Matrix
    # rows are bands and columns are dft_bins. A tall matrix.
    '''
    Example of Octave Band Matrix (obm) for 3 bands and 7 bins:
    obm = [
    [0 1 1 1 0 0 0]  # Band 1 (covers bins 1–3)
    [0 0 1 1 1 0 0]  # Band 2 (slight overlap with Band 1, bins 2–4)
    [0 0 0 1 1 1 0]  # Band 3 (covers bins 4–6)
    ]
    Overlaps are possible in some 1/3 octave cases. I assume it depends on the choosing of lowest center frequency.
    '''

    np.set_printoptions(suppress=True, formatter={'all': lambda x: f'{x:.6f}'})
    print(obm.shape, obm)

    for band in range(len(cfs)):
        dft_bin = np.argmin(np.square(dft_bins - lbe[band])) # index of smallest value/distance
        # distance between every DFT bin and a lower band edge at a time
        lbe[band] = dft_bins[dft_bin] # update lower band edge to the nearest DFT bin
        dft_bin_l_idx = dft_bin # store index of lowest distance

        dft_bin = np.argmin(np.square(dft_bins - ube[band])) # index of smallest distance
        ube[band] = dft_bins[dft_bin] # update upper band edge to the nearest DFT bin
        dft_bin_h_idx = dft_bin # store index of highest distance

        # Assign to the octave band matrix
        obm[band, dft_bin_l_idx:dft_bin_h_idx] = 1

    print(obm.shape, obm)

    # Set NumPy print options to display numbers without scientific notation
    print("Center Frequencies: ", cfs, len(cfs))
    print("Upper Band Edges: ", ube, len(ube))
    print("Lower Band Edges: ", lbe, len(lbe))

    return obm, cfs


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