# ------------------------------------------------------------------------
# This script is used to test functions while developing the mystoi module
# ------------------------------------------------------------------------

import mystoi
from pathlib import Path
import soundfile as sf
from scipy.signal import resample

# Testing third octaves function 
# mystoi.one_third_octaves(mystoi.SR, mystoi.FRAME_LEN, mystoi.OCTAVE_BANDS, mystoi.LCF)



#Load files
clean_filename = Path(r'C:\Users\George\Desktop\Automatic Control and Robotics\Semester 7\Thesis\Datasets and other Downloads\clarity_CPC1_data.test.v1\clarity_CPC1_data\clarity_data\scenes\S08569_target_anechoic.wav')
clean_audio, clean_sr = sf.read(clean_filename)
spin_filename = Path(r'C:\Users\George\Desktop\Automatic Control and Robotics\Semester 7\Thesis\Datasets and other Downloads\clarity_CPC1_data.test.v1\clarity_CPC1_data\clarity_data\HA_outputs\test\S08569_L0219_E001.wav')
spin_audio, spin_sr = sf.read(spin_filename)


# Resample
new_sr = 10000
spin_audio = resample(spin_audio, int(len(spin_audio) * new_sr / spin_sr))
clean_audio = resample(clean_audio, int(len(clean_audio) * new_sr / clean_sr))


# mystoi.remove_silent_frames(clean_filename, spin_filename, mystoi.ENERGY_RANGE, mystoi.FRAME_LEN, mystoi.OVERLAP)

# Testing shapes of audio
print(clean_audio.shape, spin_audio.shape)
clean_audio = clean_audio[:, 0]
spin_audio = spin_audio[:, 0]
print(clean_audio.shape, spin_audio.shape)

mystoi.compute_stoi(clean_audio, spin_audio, new_sr)

