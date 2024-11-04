import numpy as np
import soundfile as sf
import torch

# Function to compute third-octave filter bank
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

# Read the audio file
filename = "C:/Users/Codeexia/FinalSemester/CPC1 Data/clarity_CPC1_data.v1_1/clarity_CPC1_data/clarity_data/scenes/S08508_target_CH2.wav"
x, sr = sf.read(filename)

# Separate the left and right ear signals
le = x[:, 0]  # Left ear
re = x[:, 1]  # Right ear

# Generate the third-octave filter bank
A, cf = thirdoct(16000, 512, 15, 150)
A = torch.tensor(A)

# Function to compute STFT and similarity score
def compute_similarity(signal):
    signal_tensor = torch.tensor(signal)

    # Generate noisy signal
    noisy_signal = signal_tensor + 0.1 * torch.randn(*signal_tensor.shape)

    # Compute STFT for original and noisy signals
    window = torch.hann_window(400)
    X = torch.stft(signal_tensor, 512, hop_length=160, win_length=400, window=window, return_complex=True)
    X = torch.abs(X) ** 2  # Power spectrum
    X = X[:256, :]  # Retain only the positive frequencies

    Xd = torch.stft(noisy_signal, 512, hop_length=160, win_length=400, window=window, return_complex=True)
    Xd = torch.abs(Xd) ** 2 
    Xd = Xd[:256, :]

    Y = torch.sqrt(torch.matmul(A, X))
    Yd = torch.sqrt(torch.matmul(A, Xd))
    
    # Sliding window similarity analysis
    T = 30  # Window size
    N = Y.shape[1]  # Number of time frames
    N_prime = N - T + 1  # Number of sliding window positions
    
    ds = torch.zeros(N_prime)  # Pre-allocate tensor for storing similarity scores

    for pos in range(N_prime):
        Y_seg = Y[:, pos:(pos + T)]
        Yd_seg = Yd[:, pos:(pos + T)]

        # Normalize segments
        Y_seg = (Y_seg - torch.mean(Y_seg, dim=1, keepdim=True))
        Yd_seg = (Yd_seg - torch.mean(Yd_seg, dim=1, keepdim=True))

        # Normalize
        Y_seg = Y_seg / (torch.sqrt(torch.sum(Y_seg**2.0, dim=1, keepdim=True) + 1e-10))
        Yd_seg = Yd_seg / (torch.sqrt(torch.sum(Yd_seg**2.0, dim=1, keepdim=True) + 1e-10))

        # Compute similarity using dot product
        ds[pos] = torch.mean(torch.sum(Y_seg * Yd_seg, dim=1))

    # Calculate average similarity score for this ear
    avg_similarity = torch.mean(ds).item()
    
    return avg_similarity

# Compute similarity scores for both ears
left_similarity = compute_similarity(le)
right_similarity = compute_similarity(re)

# Choose the ear with the higher similarity score
if left_similarity > right_similarity:
    chosen_ear = 'Left Ear'
    chosen_signal = le
else:
    chosen_ear = 'Right Ear'
    chosen_signal = re

# Print the chosen ear and its similarity score
print(f"Chosen Ear for STOI Calculation: {chosen_ear}")
print(f"Left Ear Similarity Score: {left_similarity}")
print(f"Right Ear Similarity Score: {right_similarity}")
