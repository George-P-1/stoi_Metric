# Import libraries
import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import soundfile as sf
from scipy.signal import resample
import sounddevice as sd
import pystoi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

import plotter
import mystoi


# Generate timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS

# SECTION Main code here

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(path_data: DictConfig) -> None:
    # SECTION - Open reference JSON file
    try:
        with open(path_data.test_path.ref_file, 'r') as ref_file:
            ref_json = json.load(ref_file)
            
            # SECTION - Loop over each scene in JSON file and compute STOI
            # Store STOI values for mystoi
            mystoi_scores = []  # List to hold STOI values
            pystoi_scores = []  # List to hold pySTOI values
            for scene_index in range(len(ref_json)):
                # scene_index = 1006 # for running only one scene
            
                # SECTION - Load and resample SPIN and clean audio 
                # Path of audio files to open, HA_Output and target_anechoic
                spin_file_path = Path(path_data.test_path.spin_folder) / f"{ref_json[scene_index]['signal']}.wav"
                target_file_path = Path(path_data.test_path.scenes_folder) / f"{ref_json[scene_index]['scene']}_target_anechoic.wav"
                # Opening audio files using soundfile
                spin, spin_sr = sf.read(spin_file_path)
                target, target_sr = sf.read(target_file_path)
                # Resampling
                new_sr = path_data.sample_rate
                # REVIEW - Can use scipy (resample or decimate functions) or librosa library. Some issue with scipy using only frequency domain or something like that.
                spin_resampled = resample(spin, int(len(spin) * new_sr / spin_sr))   # NOTE - current_no_of_samples / current_sampling_rate is the duration of audio signal
                target_resampled = resample(target, int(len(target) * new_sr / target_sr))
                # Padding to make both signals of same length in case of different lengths
                # Pad the shorter signal
                if len(spin_resampled) < len(target_resampled): # pad spin
                    spin_resampled = np.pad(spin_resampled, (0, len(target_resampled) - len(spin_resampled)))
                elif len(target_resampled) < len(spin_resampled): # pad target
                    target_resampled = np.pad(target_resampled, (0, len(spin_resampled) - len(target_resampled)))
                    print("Padding Target audio file.") # REVIEW - REMOVE_LATER maybe
                # !SECTION
                
                # SECTION - Convert stereo to mono and plot spectrograms and play audio
                # NOTE - Convert stereo to mono [Instead of computing stoi of each ear separately]
                # REVIEW - Not sure which channel is right ear and which one is left ear.
                if len(spin.shape) == 2:
                    spin_mono = spin.mean(axis=1)
                    spin_resampled_mono = spin_resampled.mean(axis=1)
                else:
                    raise Exception("SPIN audio is not stereo.")
                if len(target.shape) == 2:
                    target_mono = target.mean(axis=1)
                    target_resampled_mono = target_resampled.mean(axis=1)
                else:
                    raise Exception("Target audio is not stereo.")

                # NOTE - Plots
                if False:    # OPTIONAL
                    # Spectrograms
                    plt.figure("Spectrograms (Mono Signals)")
                    # Plot before resampling
                    plt.subplot(2,2,1)
                    plotter.plot_spectrogram(spin_mono, spin_sr, 'Spectrogram of SPIN (before resampling)')
                    plt.subplot(2,2,2)
                    plotter.plot_spectrogram(target_mono, target_sr, 'Spectrogram of Target (before resampling)')
                    # Plot after resampling
                    plt.subplot(2,2,3)
                    plotter.plot_spectrogram(spin_resampled_mono, new_sr, 'Spectrogram of SPIN (after resampling)')
                    plt.subplot(2,2,4)
                    plotter.plot_spectrogram(target_resampled_mono, new_sr, 'Spectrogram of Target (after resampling)')
                    plt.tight_layout()  # Adjust spacing

                    # Signals in amplitude-time
                    plt.figure("Amplitude-Time (Mono Signals)")
                    # Plot before resampling
                    plt.subplot(2,2,1)
                    plotter.plot_regular(spin_mono, len(spin_mono), spin_sr, 'SPIN (before resampling)')
                    plt.subplot(2,2,2)
                    plotter.plot_regular(target_mono, len(target_mono), target_sr, 'Target (before resampling)')
                    # Plot after resampling
                    plt.subplot(2,2,3)
                    plotter.plot_regular(spin_resampled_mono, len(spin_resampled_mono), new_sr, 'SPIN (after resampling)')
                    plt.subplot(2,2,4)
                    plotter.plot_regular(target_resampled_mono, len(target_resampled_mono), new_sr, 'Target (after resampling)')
                    plt.tight_layout()  # Adjust spacing

                    plt.show()

                # NOTE - Play audio files (before and after resampling)
                if False:     # OPTIONAL
                    # Before
                    print("Playing SPIN and target audio files before resampling...")
                    sd.play(spin, spin_sr)
                    sd.wait()
                    sd.play(target, target_sr)
                    sd.wait()
                    # After
                    print("Playing SPIN and target audio files after resampling...")
                    sd.play(spin_resampled, new_sr)
                    sd.wait()   
                    sd.play(target_resampled, new_sr)
                    sd.wait()
                    # Stereo
                    print("Playing target left and right ear audio files after resampling...")
                    sd.play(target_resampled[:,0], new_sr)
                    sd.wait()
                    sd.play(target_resampled[:,1], new_sr)
                    sd.wait()
                    # Mono
                    print("Playing SPIN and target audio files after converting to mono...\n")
                    sd.play(spin_resampled_mono, new_sr)
                    sd.wait()
                    sd.play(target_resampled_mono, new_sr)
                    sd.wait()

                # !SECTION - End Loop

                # SECTION - Compute STOI Metric using pystoi
                # REMOVE_LATER
                pystoi_val = pystoi.stoi(target_resampled_mono, spin_resampled_mono, new_sr)
                # !SECTION

                # SECTION - Implement STOI Metric
                stoi_val = mystoi.compute_stoi(target_resampled_mono, spin_resampled_mono, new_sr)
                # !SECTION

                # Store computed STOI values to list
                mystoi_scores.append(stoi_val)
                pystoi_scores.append(pystoi_val)

                # break  # for running only one scene
            #!SECTION

        print(f"Number of scenes processed: {len(mystoi_scores)}")

        ref_file.close()
    except FileNotFoundError:
        raise Exception(f'JSON file not found: {path_data.test_path.ref_file}')
    finally:
        print(f'Finished processing Test JSON file.')
    #!SECTION

    # SECTION - Save mystoi_scores and true_scores to a csv file
    # TODO - Save mystoi_scores and true_scores to a csv file and calculate RMSE in different script
    mystoi_output_file = f"mystoi_scores_{timestamp}.csv"
    mystoi_df = pd.DataFrame({"Scene_Index": range(len(mystoi_scores)), "STOI_Value": mystoi_scores})
    mystoi_df.to_csv(mystoi_output_file, index=False)  # Save without the index column
    print(f"mystoi_scores saved to {mystoi_output_file}")
    # Save pystoi_scores to another CSV file
    pystoi_output_file = f"pystoi_scores_{timestamp}.csv"
    pystoi_df = pd.DataFrame({"Scene_Index": range(len(pystoi_scores)), "STOI_Value": pystoi_scores})
    pystoi_df.to_csv(pystoi_output_file, index=False)  # Save without the index column
    print(f"pystoi_scores saved to {pystoi_output_file}")
    #!SECTION


# !SECTION Main code ends here

if __name__ == '__main__':
    main()