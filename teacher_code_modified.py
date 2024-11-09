import numpy as np
import soundfile as sf
import torch
from pystoi.stoi import stoi

# Paths to the reference and degraded audio files
filename_ref = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/clarity_data/scenes/S08508_target_CH0.wav"
filename_deg = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/clarity_data/scenes/S08508_interferer_CH2.wav"
def thirdoct(fs, N_fft, num_bands, mn):
    f = np.linspace(0, fs, N_fft + 1)
    f = f[:(N_fft // 2)]
    k = np.arange(num_bands)
    cf = 2.0 ** (k / 3.0) * float(mn)
    fl = np.sqrt((2.0 ** (k / 3.0) * mn) * 2.0 ** ((k - 1) / 3.0) * float(mn))
    fr = np.sqrt((2.0 ** (k / 3.0) * mn) * 2.0 ** ((k + 1) / 3.0) * float(mn))
    
    A = np.zeros((num_bands, len(f)))
    
    for i in range(len(cf)):
        b = np.argmin((f - fl[i]) ** 2.0)
        fl[i] = f[b]
        fl_ii = b

        b = np.argmin((f - fr[i]) ** 2.0)
        fr[i] = f[b]
        fr_ii = b
        A[i, fl_ii:fr_ii] = 1.0
    
    return A, cf

# Read the reference and degraded audio files
x_ref, sr = sf.read(filename_ref)
x_deg, _ = sf.read(filename_deg)

# Separate left and right ear signals
le_ref, re_ref = x_ref[:, 0], x_ref[:, 1]  # Reference
le_deg, re_deg = x_deg[:, 0], x_deg[:, 1]  # Degraded

# Generate the third-octave filter bank
A, cf = thirdoct(10000, 512, 15, 150)
A = torch.tensor(A)

# Function to compute similarity score for a pair of signals
# # Function to compute similarity score for a pair of signals
# def compute_similarity(ref_signal, deg_signal):
#     ref_tensor = torch.tensor(ref_signal)
#     deg_tensor = torch.tensor(deg_signal)

#     # STFT for both signals
#     window = torch.hann_window(400)
#     X = torch.stft(ref_tensor, 512, hop_length=160, win_length=400, window=window, return_complex=True)
#     Xd = torch.stft(deg_tensor, 512, hop_length=160, win_length=400, window=window, return_complex=True)
#     X, Xd = torch.abs(X) ** 2, torch.abs(Xd) ** 2  # Power spectrum
#     X, Xd = X[:256, :], Xd[:256, :]  # Positive frequencies

#     Y, Yd = torch.sqrt(torch.matmul(A, X)), torch.sqrt(torch.matmul(A, Xd))
    
#     # Sliding window similarity analysis
#     T = 30
#     N, N_prime = Y.shape[1], Y.shape[1] - T + 1
    
#     ds = torch.zeros(N_prime)

#     for pos in range(N_prime):
#         Y_seg, Yd_seg = Y[:, pos:(pos + T)], Yd[:, pos:(pos + T)]

#         Y_seg -= torch.mean(Y_seg, dim=1, keepdim=True)
#         Yd_seg -= torch.mean(Yd_seg, dim=1, keepdim=True)
        
#         Y_seg /= torch.sqrt(torch.sum(Y_seg ** 2.0, dim=1, keepdim=True) + 1e-10)
#         Yd_seg /= torch.sqrt(torch.sum(Yd_seg ** 2.0, dim=1, keepdim=True) + 1e-10)

#         ds[pos] = torch.mean(torch.sum(Y_seg * Yd_seg, dim=1))

#     # Apply 40 dB threshold for silence removal
#     max_energy = torch.max(torch.sum(Y ** 2, dim=0))
#     silence_threshold = max_energy * (10 ** (-40 / 10))
#     active_indices = torch.where(torch.sum(Y ** 2, dim=0) > silence_threshold)[0]

#     # Limit active_indices to the range of ds
#     active_indices = active_indices[active_indices < N_prime]

#     # Calculate average similarity score, ignoring silent frames
#     if active_indices.numel() > 0:
#         avg_similarity = torch.mean(ds[active_indices]).item()
#     else:
#         avg_similarity = torch.mean(ds).item()  # Fallback if all frames are silent
    
#     return avg_similarity


# # Compute similarity scores for both ears
# left_similarity = compute_similarity(le_ref, le_deg)
# right_similarity = compute_similarity(re_ref, re_deg)

# # Choose the better ear based on the similarity score
# if left_similarity > right_similarity:
#     chosen_ear = 'Left Ear'
#     ref_signal, deg_signal = le_ref, le_deg
# else:
#     chosen_ear = 'Right Ear'
#     ref_signal, deg_signal = re_ref, re_deg

# STOI computation function (based on similarity)
def compute_stoi(ref_signal, deg_signal):
    ref_tensor = torch.tensor(ref_signal)
    deg_tensor = torch.tensor(deg_signal)

    # Process with the same third-octave filtering and sliding window as above
    X = torch.stft(ref_tensor, 512, hop_length=160, win_length=400, window=torch.hann_window(400), return_complex=True)
    Xd = torch.stft(deg_tensor, 512, hop_length=160, win_length=400, window=torch.hann_window(400), return_complex=True)
    X, Xd = torch.abs(X) ** 2, torch.abs(Xd) ** 2
    Y, Yd = torch.sqrt(torch.matmul(A, X[:256, :])), torch.sqrt(torch.matmul(A, Xd[:256, :]))
    
    # Similarity-based calculation for STOI over windows
    T = 30
    N, N_prime = Y.shape[1], Y.shape[1] - T + 1
    stoi_values = torch.zeros(N_prime)

    for pos in range(N_prime):
        Y_seg, Yd_seg = Y[:, pos:(pos + T)], Yd[:, pos:(pos + T)]
        Y_seg -= torch.mean(Y_seg, dim=1, keepdim=True)
        Yd_seg -= torch.mean(Yd_seg, dim=1, keepdim=True)

        Y_seg /= torch.sqrt(torch.sum(Y_seg ** 2.0, dim=1, keepdim=True) + 1e-10)
        Yd_seg /= torch.sqrt(torch.sum(Yd_seg ** 2.0, dim=1, keepdim=True) + 1e-10)

        stoi_values[pos] = torch.mean(torch.sum(Y_seg * Yd_seg, dim=1))

    # Final STOI score, averaging over all windowed segments
    stoi_score = torch.mean(stoi_values).item()
    return stoi_score

# Perform STOI on the chosen ear
stoi_score_le = compute_stoi(le_ref, le_deg)
print(f"Left ear STOI Score: {stoi_score_le}")
stoi_score_re = compute_stoi(re_ref, re_deg)
print(f"Right ear STOI Score: {stoi_score_re}")


# Compute STOI using pystoi for both ears
pystoi_left = stoi(le_ref, le_deg, sr, extended=False)
pystoi_right = stoi(re_ref, re_deg, sr, extended=False)

print(f"PySTOI Score (Left Ear): {pystoi_left}")
print(f"PySTOI Score (Right Ear): {pystoi_right}")


