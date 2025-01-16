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

import plotter
import mystoi


# SECTION Main code here

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(path_data: DictConfig) -> None:
    # print(path_data.test_path.ref_file) # REMOVE_LATER
    # SECTION - Open reference JSON file
    try:
        with open(path_data.test_path.ref_file, 'r') as ref_file:
            ref_json = json.load(ref_file)
            
            # SECTION - Loop over each scene in JSON file and compute STOI
            # Store STOI values for mystoi
            mystoi_scores = []  # List to hold STOI values
            for scene_index in range(len(ref_json)):
                # scene_index = 1006 # REMOVE_LATER - for running only one scene
            
                # SECTION - Load and resample SPIN and clean audio 
                # Path of audio files to open, HA_Output and target_anechoic
                spin_file_path = Path(path_data.test_path.spin_folder) / f"{ref_json[scene_index]['signal']}.wav"
                target_file_path = Path(path_data.test_path.scenes_folder) / f"{ref_json[scene_index]['scene']}_target_anechoic.wav"
                # Opening audio files using soundfile
                spin, spin_sr = sf.read(spin_file_path)
                target, target_sr = sf.read(target_file_path)
                # print("Length of SPIN audio file:", len(spin)) # REMOVE_LATER
                # print("Length of target audio file:", len(target)) # REMOVE_LATER
                # Resampling
                new_sr = path_data.sample_rate
                # REVIEW - Can use scipy (resample or decimate functions) or librosa library. Some issue with scipy using only frequency domain or something like that.
                spin_resampled = resample(spin, int(len(spin) * new_sr / spin_sr))   # NOTE - current_no_of_samples / current_sampling_rate is the duration of audio signal
                target_resampled = resample(target, int(len(target) * new_sr / target_sr))
                # if True:    # REMOVE_LATER
                #     print("Sample rate of SPIN audio file:", spin_sr) # REMOVE_LATER
                #     print("Sample rate of target audio file:", target_sr) # REMOVE_LATER
                #     print("Length of resampled SPIN audio file:", len(spin_resampled)) # REMOVE_LATER
                #     print("Length of resampled target audio file:", len(target_resampled)) # REMOVE_LATER
                # Padding to make both signals of same length in case of different lengths
                # Pad the shorter signal
                if len(spin_resampled) < len(target_resampled): # pad spin
                    spin_resampled = np.pad(spin_resampled, (0, len(target_resampled) - len(spin_resampled)))
                elif len(target_resampled) < len(spin_resampled): # pad target
                    target_resampled = np.pad(target_resampled, (0, len(spin_resampled) - len(target_resampled)))
                    print("Padding Target audio file.") # REVIEW - REMOVE_LATER maybe
                # if True:    # REMOVE_LATER
                #     print("Length of resampled SPIN audio file:", len(spin_resampled)) # REMOVE_LATER
                #     print("Length of resampled target audio file:", len(target_resampled)) # REMOVE_LATER
                if False:    # REMOVE_LATER
                    print("Total number of scenes in JSON file:", len(ref_json))    # REMOVE_LATER
                    print("Info about Current SPIN File:\n", ref_json[scene_index], "\n")    # REMOVE_LATER
                    print("Paths to current audio files:\n", spin_file_path, "\n", target_file_path, "\n")  # REMOVE_LATER
                    print(f"Number of samples of spin audio is {len(spin)} and sample rate is {ref_json[scene_index]['signal']} is {spin_sr} and shape is {spin.shape}.")   # REMOVE_LATER
                    print(f"Number of samples of target audio is {len(target)} and sample rate is {ref_json[scene_index]['scene']} is {target_sr} and shape is {target.shape}.")   # REMOVE_LATER
                    print(f"Number of samples of new spin audio is {len(spin_resampled)} and sample rate is {new_sr} and shape is {spin_resampled.shape}.")  # REMOVE_LATER
                    print(f"Number of samples of new target audio is {len(target_resampled)} and sample rate is {new_sr} and shape is {target_resampled.shape}.")  # REMOVE_LATER
                    # Total number of scenes in JSON file: 2421
                    # Info about Current SPIN File:
                    # {'prompt': "at home indoors i didn't ask my mum", 'scene': 'S08564', 'n_words': 8, 'listener': 'L0212', 'system': 'E001', 'volume': 50, 'signal': 'S08564_L0212_E001'}
                    # Paths to current audio files:
                    # C:\Users\George\Desktop\Automatic Control and Robotics\Semester 7\Thesis\Datasets and other Downloads\clarity_CPC1_data.test.v1\clarity_CPC1_data\clarity_data\HA_outputs\test\S08564_L0212_E001.wav
                    # C:\Users\George\Desktop\Automatic Control and Robotics\Semester 7\Thesis\Datasets and other Downloads\clarity_CPC1_data.test.v1\clarity_CPC1_data\clarity_data\scenes\S08564_target_anechoic.wav
                    # Number of samples of spin audio is 223361 and sample rate is S08564_L0212_E001 is 32000 and shape is (223361, 2).
                    # Number of samples of target audio is 307818 and sample rate is S08564 is 44100 and shape is (307818, 2).
                    # Number of samples of new spin audio is 69800 and sample rate is 10000 and shape is (69800, 2).
                    # Number of samples of new target audio is 69800 and sample rate is 10000 and shape is (69800, 2).
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
                stoi_val = pystoi.stoi(target_resampled_mono, spin_resampled_mono, new_sr)
                # print(f"STOI value for intelligibility of Mono SPIN signal {ref_json[scene_index]['signal']} is {stoi_val}.") # REMOVE_LATER
                # !SECTION

                # SECTION - Implement STOI Metric
                # stoi_val = mystoi.compute_stoi(target_resampled_mono, spin_resampled_mono, new_sr)
                # print("----------mystoi values below----------\n") # REMOVE_LATER
                # print(f"STOI value for intelligibility of Mono SPIN signal {ref_json[scene_index]['signal']} is {stoi_val}.") # REMOVE_LATER
                # !SECTION

                # Store STOI values for mystoi
                mystoi_scores.append(stoi_val)
                print("current scene index:", scene_index) # REMOVE_LATER
                # break  # REMOVE_LATER - for running only one scene
            #!SECTION

        print(f"Number of scenes processed: {len(mystoi_scores)}") # REMOVE_LATER

        ref_file.close()
    except FileNotFoundError:
        raise Exception(f'JSON file not found: {path_data.test_path.ref_file}')
    finally:
        print(f'Finished processing Test JSON file.')
    #!SECTION

    # SECTION - Open result JSON file
    try:
        with open(path_data.test_result_path.result_ref_file, 'r') as res_file:
            res_json = json.load(res_file)

            # Add correctness scores to a list
            true_scores = []

            for res_index in range(len(res_json)):
                true_scores.append(res_json[res_index]['correctness'])

    except FileNotFoundError:
        raise Exception(f'JSON file not found: {path_data.test_result_path.result_ref_file}')
    finally:
        print(f'Finished processing Result JSON file.')
    #!SECTION

    # SECTION - Calculate RMSE
    # TODO - Save mystoi_scores and true_scores to a csv file and calculate RMSE in different script
    rmse_val = mystoi.calc_RMSE(np.array(mystoi_scores), np.array(true_scores))
    print(f"RMSE value for STOI metric is {rmse_val}.")
    # !SECTION

# !SECTION Main code ends here

if __name__ == '__main__':
    main()