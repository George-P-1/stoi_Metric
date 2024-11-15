import numpy as np
import torch

def compute_stoi(ref_signal, deg_signal, fs=10000, n_fft=512, num_bands=15, min_freq=150):
    T = 30  # Window length in frames

    # Third-Octave Filter Bank Generation
    def thirdoct(fs, N_fft, num_bands, mn):
        f = np.linspace(0, fs, N_fft + 1)[:N_fft // 2]
        k = np.arange(num_bands)
        cf = 2.0 ** (k / 3.0) * mn
        fl = np.sqrt((2.0 ** (k / 3.0) * mn) * 2.0 ** ((k - 1) / 3.0) * mn)
        fr = np.sqrt((2.0 ** (k / 3.0) * mn) * 2.0 ** ((k + 1) / 3.0) * mn)
        
        A = np.zeros((num_bands, len(f)))
        for i in range(len(cf)):
            fl_ii, fr_ii = np.argmin((f - fl[i]) ** 2), np.argmin((f - fr[i]) ** 2)
            A[i, fl_ii:fr_ii] = 1.0
        return torch.tensor(A), cf

    # Generate filter bank
    A, cf = thirdoct(fs, n_fft, num_bands, min_freq)

    # Convert signals to torch tensors
    ref_tensor, deg_tensor = torch.tensor(ref_signal), torch.tensor(deg_signal)

    # STFT and Power Spectrum calculation
    X = torch.stft(ref_tensor, n_fft, hop_length=160, win_length=400, window=torch.hann_window(400), return_complex=True)
    Xd = torch.stft(deg_tensor, n_fft, hop_length=160, win_length=400, window=torch.hann_window(400), return_complex=True)
    X, Xd = torch.abs(X) ** 2, torch.abs(Xd) ** 2

    # Apply Third-Octave Filtering
    Y, Yd = torch.sqrt(torch.matmul(A, X[:n_fft // 2, :])), torch.sqrt(torch.matmul(A, Xd[:n_fft // 2, :]))

#################################
    # Silent frame removal based on energy threshold
    max_energy = torch.max(torch.sum(Y ** 2, dim=0))
    silence_threshold = max_energy * (10 ** (-40 / 10))
    active_indices = torch.where(torch.sum(Y ** 2, dim=0) > silence_threshold)[0]
    
    # Select active frames for STOI computation
    Y_active, Yd_active = Y[:, active_indices], Yd[:, active_indices]

    # STOI calculation over sliding windows
    N_prime = Y_active.shape[1] - T + 1
    stoi_values = torch.zeros(N_prime)

    for pos in range(N_prime):
        Y_seg, Yd_seg = Y_active[:, pos:pos + T], Yd_active[:, pos:pos + T]
        Y_seg -= Y_seg.mean(dim=1, keepdim=True)
        Yd_seg -= Yd_seg.mean(dim=1, keepdim=True)
        Y_seg /= torch.sqrt((Y_seg ** 2).sum(dim=1, keepdim=True) + 1e-10)
        Yd_seg /= torch.sqrt((Yd_seg ** 2).sum(dim=1, keepdim=True) + 1e-10)
        stoi_values[pos] = (Y_seg * Yd_seg).sum(dim=1).mean()
    return stoi_values.mean().item()    
######################


    # N_prime = Y.shape[1] - T + 1
    # stoi_values = torch.zeros(N_prime)
    
    # for pos in range(N_prime):
    #     Y_seg, Yd_seg = Y[:, pos:pos + T], Yd[:, pos:pos + T]
    #     Y_seg -= Y_seg.mean(dim=1, keepdim=True)
    #     Yd_seg -= Yd_seg.mean(dim=1, keepdim=True)
    #     Y_seg /= torch.sqrt((Y_seg ** 2).sum(dim=1, keepdim=True) + 1e-10)
    #     Yd_seg /= torch.sqrt((Yd_seg ** 2).sum(dim=1, keepdim=True) + 1e-10)
    #     stoi_values[pos] = (Y_seg * Yd_seg).sum(dim=1).mean()
    # return stoi_values.mean().item()
