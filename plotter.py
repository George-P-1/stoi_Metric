import numpy as np
from scipy.signal import spectrogram

import matplotlib.pyplot as plt


def plot_spectrogram(audio_data, sampling_rate, title):
    f, t, Sxx = spectrogram(audio_data, fs=sampling_rate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.colorbar(label='Intensity [dB]')

def plot_regular(audio_data, num_samples, sampling_rate, title):
    duration = num_samples / sampling_rate
    sampling_time = 1 / sampling_rate
    time_axis = np.arange(0, duration, sampling_time)
    plt.plot(time_axis, audio_data)
    plt.title(title)
    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude')