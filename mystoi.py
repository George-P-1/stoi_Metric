"""
STOI
====


How to use?
-----------



Dependencies
------------
    1. Numpy
    2. Scipy


References
----------
[1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
    Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
    IEEE Transactions on Audio, Speech, and Language Processing, 2011.


"""

# Import libraries
import numpy as np
from scipy.signal import stft

# ------------------------------------
# NOTE - How to calculate STOI?
# Core steps: 
#   1. Remove Silent Frames
#   2. STFT
#   3. Convert Linear Frequency data to 1/3 Octave Bands data
#   4. Clip the spin audio based on clean audio peaks
#   5. Normalize
#   6. Compare to get intermediate intelligibility measure 
#   7. Average the intermediate intelligibility measure over all bands and frames to get final STOI value
#
# Resample audios to 10kHz
#           Now audio is in amplitude-time domain
# TF-decompose the signals, i.e. make them in time frequency domain. using moving Hann window for both clean and spin audio. (Hann, 256 samples, 512, 50%, stft)
#           When zero padding the frame (done within stft function), all the padding can be done after the the 256 length frame. Instead of 128 on both sides.
#           So this means that, u choose 256 samples from the signal and then apply Hann window and then do stft. Then choose the next frame of 256 samples which overlaps 128 samples with the previous frame. 
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
FRAME_LEN = 512             # =256 + OVERLAP * 2. This frame length is used for STFT
ENERGY_RANGE = 40           # Speech dynamic range
# one-third octave related
OCTAVE_BANDS = 15           # Number of one-third octave bands used
LCF = 150                   # lowest center frequency. Estimates higher c.f. is approx. 4.3kHz
# Comparison related
FRAME_TIME = 384            # 386 or 384 milliseconds. Paper shows both numbers in first few pages.
ANALYSIS_FRAME_LEN = 30     # Number of frames which equals an analysis window of 384ms
EPS = np.finfo("float").eps # Epsilon, smallest positive float number.

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
    if SR != sampling_rate:
        raise Exception("Sampling rate is not {}".format(SR))

    # TODO - NOTE - Remove silent frames
    # clean_audio, spin_audio = remove_silent_frames(clean_audio, spin_audio, ENERGY_RANGE, TRUE_FRAME_LEN, OVERLAP)

    # TF Decomposition
    # NOTE - Short-Time Fourier Transform on both signals - to obtain DFT bins
    f_clean, t_clean, clean_stft = stft(clean_audio, sampling_rate, 'hann', TRUE_FRAME_LEN, OVERLAP, FRAME_LEN)
    f_spin, t_spin, spin_stft = stft(spin_audio, sampling_rate, 'hann', TRUE_FRAME_LEN, OVERLAP, FRAME_LEN)

    # Get the one-third octave band matrix and center frequencies
    obm, cfs = one_third_octaves(sampling_rate, FRAME_LEN, OCTAVE_BANDS, LCF)

    # NOTE - Group DFT bins into one-third octave bands
    # Using Equation 1 from the paper - Apply OBM boolean matrix to spectrogram (STFT)
    clean_tf_units = np.sqrt(np.matmul(obm, (np.square(np.abs(clean_stft)))))
    spin_tf_units = np.sqrt(np.matmul(obm, (np.square(np.abs(spin_stft)))))

    if False:#REMOVE_LATER - Print shapes
        print("Clean audio shape: {} and spin audio shape: {}".format(clean_audio.shape, spin_audio.shape))
        # Clean audio shape: (61400,) and spin audio shape: (61400,)
        print("Octave Band Matrix shape: {} and Center frequencies vector shape: {}".format(obm.shape, cfs.shape))
        # Octave Band Matrix shape: (15, 257) and Center frequencies vector shape: (15,)
        # REVIEW - 15 is the number of octave bands and 257 is the number of DFT bins. So there are 257 DFT bins in each octave band.
        print("Clean_stft shape: {} and spin_stft shape: {}".format(clean_stft.shape, spin_stft.shape))
        # Clean_stft shape: (257, 481) and spin_stft shape: (257, 481)
        # REVIEW - 257 is the number of DFT bins and 481 is the length of time axis. So there is a spectrogram of 257x481 for this (scene_index=4) signal and so each point in time has 257 DFT bins.
        # NOTE - 481 is the number of frames taken from the time axis of the audio signal. So fourier transform is done on 481 frames of the audio signal.
        print("Clean_tf_units shape: {} and spin_tf_units shape: {}".format(clean_tf_units.shape, spin_tf_units.shape))
        # Clean_tf_units shape: (15, 481) and spin_tf_units shape: (15, 481)
        # REVIEW - 15 is the number of octave bands and 481 is the length of time axis.
        # NOTE !!!! - The value 481 (the second dimension of the tf_units matrix) depends on the audio signal. 481 is when scene_index = 4.
    
    # NOTE - Create 3D matrices of temporal envelopes (analysis windows) for clean and spin audio
    # Use a bigger frame of 30 frames (within 481) and 15 bands which equals an analysis window of 384ms
    # Use Equation 2 from the paper - short-time temporal envelope
    clean_an_windows = []
    spin_an_windows = []
    for m in range(ANALYSIS_FRAME_LEN, clean_tf_units.shape[1] + 1): # Go through the length of time axis using frame. The +1 is to include the last point since for example, range(1, 10) doesn't include 10.
        clean_an_win = clean_tf_units[:, m - ANALYSIS_FRAME_LEN : m]  # Analysis window for clean audio
        spin_an_win = spin_tf_units[:, m - ANALYSIS_FRAME_LEN : m]   # Analysis window for spin audio  
        # There is no +1 like in Equation 2 like m - ANALYSIS_FRAME_LEN +1: m, because the index 'm' is not included in the notation [0:m] so [0:30] would have 30 elements not 31.  
        clean_an_windows.append(clean_an_win)
        spin_an_windows.append(spin_an_win)
        if False: # REMOVE_LATER
            print("index of start of big frame: {}".format(m - ANALYSIS_FRAME_LEN)) # 0 at start of loop and at end 451
            print("total iterations of loop: {}".format(len(range(ANALYSIS_FRAME_LEN, clean_tf_units.shape[1] + 1)))) # 452
            print("m: {}".format(m)) # at start of loop its 30, and ends at 481
            print("Clean analysis window shape: {} and spin analysis window shape: {}".format(clean_an_win.shape, spin_an_win.shape))
            # Clean analysis window shape: (15, 30) and spin analysis window shape: (15, 30)
    clean_an_windows = np.array(clean_an_windows)
    spin_an_windows = np.array(spin_an_windows)

    if True: # REMOVE_LATER - Print shapes
        print("Clean analysis windows shape: {} and spin analysis windows shape: {}".format(clean_an_windows.shape, spin_an_windows.shape))
        # Clean analysis windows shape: (452, 15, 30) and spin analysis windows shape: (452, 15, 30)
    
    # TODO - NOTE - Clip the spin audio based on clean audio peaks

    # TODO - NOTE - Normalize

    # TODO - Compare to get intermediate intelligibility measure

    # TODO - Average the intermediate intelligibility measure over all bands and frames to get final STOI value

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

    """ # REMOVE_LATER
    Example of Octave Band Matrix (obm) for 3 bands and 7 bins:
    obm = [
    [0 1 1 1 0 0 0]  # Band 1 (covers bins 1–3)
    [0 0 1 1 1 0 0]  # Band 2 (slight overlap with Band 1, bins 2–4)
    [0 0 0 1 1 1 0]  # Band 3 (covers bins 4–6)
    ]
    Overlaps are possible in some 1/3 octave cases. I assume it depends on the choosing of lowest center frequency.
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
    # REVIEW - The paper mentions highest center frequency is approx. 4.3kHz but it isn't when calculated.
    ube = lcf * np.power(2.0, (2 * bands + 1) / (2*octave_kind))  # upper band edge
    # NOTE - However, the upper band edge of highest band is approx. 4.3kHz
    lbe = lcf * np.power(2.0, (2 * bands - 1) / (2*octave_kind))  # lower band edge
    
    obm = np.zeros((num_bands, len(dft_bins)))  # Octave Band Matrix
    # rows are bands and columns are dft_bins. A tall matrix.

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

    return obm, cfs


# STUB
def remove_silent_frames(clean_audio, spin_audio, dyn_range, true_frame_len, overlap):
    """
    Removes silent frames from clean and spin audio.
    Removes all frames in both signals in which the energy in clean speech signal is lower than 40dB.

    Arguments:
        clean_audio: clean speech (target_anechoic)
        spin_audio: Speech-in-Noise
        dyn_range (int): Speech dynamic range
        frame_len (int): Number of samples in a frame of signal
        overlap (int): Number of samples in overlap

    Returns:
        (clean_audio, spin_audio) (np.ndarray, np.ndarray): 
        Clean and Spin audio after removing silent frames
    """
    # Create Hann window mask
    hann_window = np.hanning(true_frame_len) # REVIEW - Should it be frame_len + 2? np.hanning(framelen + 2)[1:-1]?
    
    # Create an array of frames for both clean and spin audio
    clean_frames = []
    spin_frames = []
    for i in range(0, len(clean_audio) - true_frame_len, overlap): # 0 128 256 384 ...
        clean_frames.append(hann_window * clean_audio[i : i+true_frame_len])
        spin_frames.append(hann_window * spin_audio[i : i+true_frame_len])
    # Convert list to np.ndarray
    clean_frames = np.array(clean_frames)
    spin_frames = np.array(spin_frames)

    # Compute energy in dB for clean audio frames
    clean_energies = 20 * np.log10(np.linalg.norm(clean_frames, axis=1) + EPS)
    # The 20 * log10 is to convert energy to dB. axis=1 is to compute energy of each frame.
    # The norm is to compute energy of each frame. The EPS is to prevent log(0).    

    # Create mask to remove silent frames
    mask = clean_energies > np.max(clean_energies) - dyn_range  # REVIEW - Make sure inequality is correct

    # Use mask on both clean and spin audio frames
    clean_frames = clean_frames[mask]
    spin_frames = spin_frames[mask]

    # TODO - Reconstruct clean and spin audio


    pass
    # return clean_audio, spin_audio


def calc_RMSE(stoi_arr, listeners_arr) -> float:
    """
    Calculates Root Mean Squared Error value when given a list of computed STOI values and list of actual listeners scores (true intelligibility). 
    
    Arguments:
        stoi_arr (list[float] or np.ndarray): A list of STOI values
        listeners_arr (list[float] or np.ndarray): A list of true intelligibility values from clarity JSON file

    Returns:
        float: RMSE value
    """
    error = stoi_arr - listeners_arr
    return np.sqrt(np.mean((error) ** 2))